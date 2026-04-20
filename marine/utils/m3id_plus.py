import torch
import numpy as np
from PIL import Image
import requests
import torch.nn as nn
from io import BytesIO
import cv2
import matplotlib.pyplot as plt
import copy
from tqdm.auto import tqdm
import seaborn as sns
import torch.nn.functional as F
from transformers import AutoProcessor, LlavaForConditionalGeneration

def calculate_spatial_entropy(att_map):
    """
    Tính Spatial Entropy cho một attention map.
    Dựa trên thuật toán từ paper Kang et al. (2025).
    """
    # 1. Binarize: Gán 1 cho các phần tử trên trung bình, 0 cho phần tử dưới trung bình
    heatmap = att_map.detach().cpu().to(torch.float32).numpy()
    mean_val = heatmap.mean()
    binary_map = (heatmap > mean_val).astype(np.uint8)

    # 2. Tìm các connected components (8-neighbors)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)

    if num_labels <= 1: # Không tìm thấy cụm nào
        return torch.tensor([1.0], device=att_map.device, dtype=torch.float32)

    # 3. Tính P(Ci) dựa trên diện tích các cụm (bỏ qua background label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    total_area = areas.sum()
    if total_area == 0:
        return torch.tensor([1.0], device=att_map.device, dtype=torch.float32)

    p_ci = areas / total_area

    # 4. Tính toán Entropy
    entropy = -np.sum(p_ci * np.log(p_ci + 1e-9))

    # Normalize entropy về khoảng [0, 1]
    normalized_entropy = entropy / np.log(num_labels) if num_labels > 1 else 1.0
    return torch.tensor([normalized_entropy], device=att_map.device, dtype=torch.float32)


class GammaNet(nn.Module):
    def __init__(self, hidden_dim=4096):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_dim + 1) # +1 cho giá trị Entropy
        self.gamma_layer = nn.Sequential(
            nn.Linear(hidden_dim + 1, (hidden_dim + 1) // 4),
            nn.GELU(),
            nn.Linear((hidden_dim + 1) // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, h_t, entropy_t):
        combined_input = torch.cat([h_t, entropy_t.view(-1, 1)], dim=-1)
        combined_input = self.input_norm(combined_input)
        return self.gamma_layer(combined_input)


class LLaVA_M3ID_Plus:
    def __init__(
        self,
        model,
        processor,
        gamma_net,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.processor = processor
        # FIX: Ensure GammaNet is forced to float32 to prevent FP16 LayerNorm overflow/NaNs
        self.gamma_net = gamma_net.to(device).to(torch.float32).eval()
        self.device = device

        self.loc_layer = [24, 26]
        self.loc_heads = [15, 17, 21]

    def load_image(self, image_source) -> Image.Image:
        if isinstance(image_source, Image.Image):
            return image_source
        if image_source.startswith(('http://', 'https://')):
            response = requests.get(image_source)
            return Image.open(BytesIO(response.content)).convert('RGB')
        return Image.open(image_source).convert('RGB')

    def shuffle_patches(self, image: Image.Image, patch_size: int = 14) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")

        w, h = image.size
        new_w, new_h = (w // patch_size) * patch_size, (h // patch_size) * patch_size
        if (w, h) != (new_w, new_h):
            image = image.resize((new_w, new_h), resample=Image.LANCZOS)

        img_array = np.array(image)
        h, w, c = img_array.shape

        patches = img_array.reshape(h // patch_size, patch_size, w // patch_size, patch_size, c)
        patches = patches.swapaxes(1, 2).reshape(-1, patch_size, patch_size, c)
        indices = np.random.permutation(len(patches))
        shuffled_patches = patches[indices]

        shuffled_img = shuffled_patches.reshape(h // patch_size, w // patch_size, patch_size, patch_size, c)
        shuffled_img = shuffled_img.swapaxes(1, 2).reshape(h, w, c)
        return Image.fromarray(shuffled_img)

    def generate_heatmap(self, att_map, raw_image, token_text):
        num_patches = att_map.shape[-1]
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            grid_size = 24

        heatmap = att_map.mean(dim=0).view(grid_size, grid_size).detach().cpu().to(torch.float32).numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        img_array = np.array(raw_image)
        h, w = img_array.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_resized = np.uint8(255 * heatmap_resized)

        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        superimposed = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)

        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
        plt.title(f"Visual Grounding: '{token_text}'")
        plt.axis('off')
        plt.show()

    @torch.inference_mode()
    def generate(self, prompt: str, image_path: str, max_new_tokens: int = 100, visualize=True):
        raw_image = self.load_image(image_path)
        shuffled_image = self.shuffle_patches(raw_image)

        prompt_formatted = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs_c = self.processor(text=prompt_formatted, images=raw_image, return_tensors="pt").to(self.device)
        inputs_s = self.processor(text=prompt_formatted, images=None, return_tensors="pt").to(self.device)

        out_c = self.model(**inputs_c, use_cache=True, output_attentions=True, output_hidden_states=True)
        out_s = self.model(**inputs_s, use_cache=True)

        past_kv_c, past_kv_s = out_c.past_key_values, out_s.past_key_values
        logits_c, logits_s = out_c.logits[:, -1, :], out_s.logits[:, -1, :]

        input_ids = inputs_c['input_ids'][0]
        img_token_id = getattr(self.processor, "image_token_id", 32000)
        img_indices = (input_ids == img_token_id).nonzero(as_tuple=True)[0]

        if len(img_indices) == 0:
            img_start, img_end = 1, 577 # Fallback mặc định cho LLaVA-1.5
        else:
            img_start, img_end = img_indices[0].item(), img_indices[-1].item() + 1

        grid_dim = int((img_end - img_start) ** 0.5)
        generated_ids = []

        for t in range(max_new_tokens):
            # att_map_raw = out_c.attentions[self.loc_layer][:, self.loc_heads, -1, img_start:img_end]

            selected_layer_attentions = []
            for layer_idx in self.loc_layer:
                layer_attention = out_c.attentions[layer_idx]
                current_layer_att = layer_attention[:, self.loc_heads, -1, img_start:img_end]
                selected_layer_attentions.append(current_layer_att)

            att_map_raw_stacked = torch.stack(selected_layer_attentions, dim=0)

            att_map_raw = att_map_raw_stacked.mean(dim=0)


            alpha_t = att_map_raw.sum(dim=-1).mean().clamp(min=0.01, max=1.0)

            current_att_grid = att_map_raw[0].mean(dim=0).view(grid_dim, grid_dim)
            entropy_t = calculate_spatial_entropy(current_att_grid).to(self.device)

            # FIX: Cast h_t and entropy_t to float32 before feeding to GammaNet
            h_t = out_c.hidden_states[-1][:, -1, :].to(torch.float32)
            gamma_t = self.gamma_net(h_t, entropy_t.to(torch.float32)).clamp(min=0.05, max=0.95)

            lc, ls = torch.log_softmax(logits_c, dim=-1), torch.log_softmax(logits_s, dim=-1)

            # THE FIX: Add 'cd_weight' to scale the penalty.
            # If cd_weight = 1.0, it is your current code.
            # If cd_weight = 0.5, it halves the contrastive strength, restoring Precision.
            cd_weight = 0.5

            # FIX: Cast correction_weight back to LLM's precision (float16) to avoid type mismatches
            correction_weight = torch.sqrt((1 - gamma_t) / gamma_t).view(lc.size(0), 1).to(lc.dtype)
            l_star = lc + cd_weight * alpha_t * correction_weight * (lc - ls)

            probs = torch.softmax(l_star, dim=-1)
            # next_token = torch.multinomial(probs, num_samples=1)
            next_token = torch.argmax(l_star, dim=-1).unsqueeze(0)

            if next_token.item() == self.processor.tokenizer.eos_token_id:
                break

            generated_ids.append(next_token.item())
            token_text = self.processor.tokenizer.decode([next_token.item()])

            if visualize:
                print(f"Token {t}: '{token_text}' (Alpha: {alpha_t.item():.4f})")
                self.generate_heatmap(att_map_raw[0], raw_image, token_text)

            out_c, past_kv_c = self.step(next_token, past_kv_c, inputs_c)
            out_s, past_kv_s = self.step(next_token, past_kv_s, inputs_s)
            logits_c, logits_s = out_c.logits[:, -1, :], out_s.logits[:, -1, :]

        return self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def step(self, token, past_kv, original_inputs):
        if hasattr(past_kv, "get_seq_length"):
            current_seq_len = past_kv.get_seq_length()
        else:
            current_seq_len = past_kv[0][0].shape[2]

        mask = original_inputs['attention_mask']
        new_mask = torch.ones((mask.shape[0], 1), device=self.device, dtype=mask.dtype)
        original_inputs['attention_mask'] = torch.cat([mask, new_mask], dim=-1)

        outputs = self.model(
            input_ids=token.view(-1, 1),
            past_key_values=past_kv,
            attention_mask=original_inputs['attention_mask'],
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True
        )
        return outputs, outputs.past_key_values


def dpo_loss_gamma(g_chosen, g_rejected, beta=0.1):
    diff = g_rejected - g_chosen
    return -F.logsigmoid(beta * diff).mean()


def plot_m3id_plus_results(history, gamma_history):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train DPO Loss', color='blue', marker='o')
    plt.plot(history['val_loss'], label='Val DPO Loss', color='red', marker='x')
    plt.title('M3ID+ Gamma Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    chosen_vals = np.array(gamma_history['chosen'])
    rejected_vals = np.array(gamma_history['rejected'])

    sns.kdeplot(chosen_vals, fill=True, color="green", label=r"$\gamma$ Chosen (Real Objects)", bw_adjust=0.5)
    sns.kdeplot(rejected_vals, fill=True, color="orange", label=r"$\gamma$ Rejected (Hallucinations)", bw_adjust=0.5)

    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    plt.title(r'Divergence of $\gamma$ values (Final Epoch)')
    plt.xlabel(r'Gamma Value ($\gamma$)')
    plt.ylabel('Density')
    plt.legend()

    plt.tight_layout()
    plt.show()


def get_hidden_states(engine, sample):
    raw_img = engine.load_image(sample['image'])
    shuffled_img = engine.shuffle_patches(raw_img)

    prompt = sample.get('prompt', "Describe the image in detail.")
    prompt_formatted = f"USER: <image>\n{prompt}\nASSISTANT:"

    inputs_c = engine.processor(text=prompt_formatted, images=raw_img, return_tensors="pt").to(engine.device)
    inputs_s = engine.processor(text=prompt_formatted, images=shuffled_img, return_tensors="pt").to(engine.device)

    with torch.no_grad():
        out_c = engine.model(**inputs_c, output_hidden_states=True, output_attentions=True)
        out_s = engine.model(**inputs_s, output_hidden_states=True)

        # FIX: Convert hidden states to float32 to prevent GammaNet from corrupting FP16
        h_c = out_c.hidden_states[-1][:, -1, :].to(torch.float32)
        h_s = out_s.hidden_states[-1][:, -1, :].to(torch.float32)

        input_ids = inputs_c['input_ids'][0]
        img_token_id = getattr(engine.processor, "image_token_id", -200)
        if img_token_id not in input_ids and 32000 in input_ids:
            img_token_id = 32000

        img_indices = (input_ids == img_token_id).nonzero(as_tuple=True)[0]
        if len(img_indices) == 0:
            img_start, img_end = 1, 577
        else:
            img_start, img_end = img_indices[0].item(), img_indices[-1].item() + 1

        att_map = out_c.attentions[engine.loc_layer][:, engine.loc_heads, -1, img_start:img_end].mean(dim=1)

        num_patches = img_end - img_start
        grid_dim = int(num_patches ** 0.5)
        if grid_dim * grid_dim != num_patches:
            grid_dim = 24

        entropy_t = calculate_spatial_entropy(att_map[0].view(grid_dim, grid_dim))

        # FIX: Ensure entropy is float32
        return h_c, h_s, entropy_t.to(h_c.device).to(torch.float32)


def train_gamma(
    m3id_engine,
    train_samples,
    val_samples,
    epochs=10,
    lr=1e-5,
    patience=2
):
    optimizer = torch.optim.AdamW(m3id_engine.gamma_net.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_model_weights = None
    epochs_no_improve = 0

    history = {'train_loss': [], 'val_loss':[]}
    gamma_history = {'chosen': [], 'rejected':[]}

    for epoch in range(epochs):
        m3id_engine.gamma_net.train()
        total_train_loss = 0
        valid_samples = 0

        pbar = tqdm(train_samples, desc=f"Epoch {epoch+1} [Train]")

        for sample in pbar:
            optimizer.zero_grad()
            h_chosen, h_rejected, entropy_t = get_hidden_states(m3id_engine, sample)

            # Catch LLaVA extreme edge cases resulting in NaN representation
            if torch.isnan(h_chosen).any() or torch.isnan(h_rejected).any():
                continue

            g_chosen = m3id_engine.gamma_net(h_chosen, entropy_t)
            g_rejected = m3id_engine.gamma_net(h_rejected, entropy_t)

            loss = dpo_loss_gamma(g_chosen, g_rejected)

            # Fallback NaN protector mechanism
            if torch.isnan(loss):
                print("Warning: NaN loss detected! Skipping step.")
                continue

            loss.backward()

            # Gradient clipping blocks explosion caused by sparse batches
            torch.nn.utils.clip_grad_norm_(m3id_engine.gamma_net.parameters(), max_norm=1.0)

            optimizer.step()

            total_train_loss += loss.item()
            valid_samples += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / max(1, valid_samples)
        history['train_loss'].append(avg_train_loss)

        m3id_engine.gamma_net.eval()
        total_val_loss = 0
        val_valid_samples = 0

        current_epoch_gamma = {'chosen': [], 'rejected':[]}

        with torch.no_grad():
            for sample in val_samples:
                h_chosen, h_rejected, entropy_t = get_hidden_states(m3id_engine, sample)

                if torch.isnan(h_chosen).any() or torch.isnan(h_rejected).any():
                    continue

                g_chosen = m3id_engine.gamma_net(h_chosen, entropy_t)
                g_rejected = m3id_engine.gamma_net(h_rejected, entropy_t)

                v_loss = dpo_loss_gamma(g_chosen, g_rejected)

                if not torch.isnan(v_loss):
                    total_val_loss += v_loss.item()
                    val_valid_samples += 1
                    current_epoch_gamma['chosen'].append(g_chosen.item())
                    current_epoch_gamma['rejected'].append(g_rejected.item())

        avg_val_loss = total_val_loss / max(1, val_valid_samples)
        history['val_loss'].append(avg_val_loss)
        gamma_history = current_epoch_gamma

        print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(m3id_engine.gamma_net.state_dict())
            epochs_no_improve = 0
            print(" Val loss improved. Saving checkpoint...")
        else:
            epochs_no_improve += 1
            print(f" No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f" Early stopping triggered.")
            break

    m3id_engine.gamma_net.load_state_dict(best_model_weights)
    torch.save(m3id_engine.gamma_net.state_dict(), "gamma_net_weights.pth")

    print("\n--- Hoàn tất huấn luyện. Đang hiển thị kết quả phân tích ---")
    plot_m3id_plus_results(history, gamma_history)

    return history



if __name__ == "__main__":
    # Khởi tạo mô hình LLaVA-1.5-7B gốc
    model_id = "llava-hf/llava-1.5-7b-hf" # Hoặc đường dẫn local của bạn
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="eager" # Fix: Add this to enable attention output
    )

    # 1. Khởi tạo GammaNet (Đảm bảo hidden_dim khớp với LLaVA-7B)
    gamma_net = GammaNet(hidden_dim=4096)

    # 2. Load trọng số đã huấn luyện từ file .pth
    weights_path = "gamma_net_weights_full.pth"
    # gamma_net.load_state_dict(torch.load(weights_path))
    gamma_net.to("cuda").to(torch.float32).eval() # Chuyển về eval mode
    
    print(f" Đã load thành công trọng số Gamma Net")

    # 3. Khởi tạo Engine M3ID+
    llava_m3id_plus = LLaVA_M3ID_Plus(
        model=model,
        processor=processor,
        gamma_net=gamma_net,
        device="cuda",
    )
