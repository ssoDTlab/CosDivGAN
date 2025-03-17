import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchmetrics
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM

# GPU 또는 CPU 선택
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# MS-SSIM을 이용한 Mode Collapse 분석 (폴더에서 이미지 로드)
def calculate_mode_collapse_ms_ssim_from_folder(fake_dir, device, num_samples=20000, num_pairs=10000):
    # kernel_size=7, betas=(0.5, 0.5, 0.5)로 설정하여 작은 이미지 지원
    ms_ssim = MS_SSIM(data_range=1.0, kernel_size=7, betas=(0.5, 0.5, 0.5)).to(device)

    # 1️⃣ fake_images 폴더에서 이미지 리스트 가져오기
    fake_images_list = sorted(os.listdir(fake_dir))[:num_samples]  # 20,000장 선택
    transform = transforms.Compose([transforms.ToTensor()])  # 이미지를 Tensor로 변환

    # 2️⃣ 이미지를 Tensor로 변환하여 리스트에 저장
    fake_images = []
    for img_name in fake_images_list:
        img_path = os.path.join(fake_dir, img_name)
        img = Image.open(img_path).convert("RGB")  # RGB 변환
        img = transform(img).to(device)  # Tensor 변환 후 GPU로 이동
        fake_images.append(img)

    # 3️⃣ Tensor로 변환 (N, C, H, W) 형태
    fake_images = torch.stack(fake_images)  # (num_samples, 3, 64, 64)

    # 4️⃣ MS-SSIM 계산 (num_pairs만큼 샘플링하여 비교)
    num_samples = fake_images.shape[0]
    total_score = 0.0
    num_pairs = min(num_pairs, num_samples)  # 샘플 개수 초과 방지

    for _ in range(num_pairs):
        idx1, idx2 = torch.randint(0, num_samples, (2,))
        img1 = fake_images[idx1].unsqueeze(0)
        img2 = fake_images[idx2].unsqueeze(0)

        score = ms_ssim(img1, img2)
        total_score += score.item()

    return total_score / num_pairs


# 5️⃣ MS-SSIM 측정 (폴더에서 10,000장 불러와 5,000쌍 비교)
fake_dir = "../DCGAN_fake_images"
ms_ssim_score = calculate_mode_collapse_ms_ssim_from_folder(fake_dir, device, num_samples=20000, num_pairs=10000)

print(f"Mode Collapse MS-SSIM Score (10000 pairs, from folder): {ms_ssim_score:.4f}")
