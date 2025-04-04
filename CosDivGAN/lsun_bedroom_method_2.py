import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.nn.functional as F
from cleanfid import fid


# 학습 완료 후 최종 모델 저장
def save_final_model(netG, netD, fid_score=None, epoch_setting=None):
    save_dir = "../saved_models"
    os.makedirs(save_dir, exist_ok=True)

    # 에폭 설정에 따른 모델 경로
    model_path = os.path.join(save_dir, f"CosDivGAN_epoch_{epoch_setting}.pth")

    # 모델 상태 및 주요 정보 저장
    checkpoint = {
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'fid_score': fid_score,
        'epoch_setting': epoch_setting
    }

    # 모델 저장
    torch.save(checkpoint, model_path)
    print(f"최종 모델 저장 완료: {model_path}")
    if fid_score is not None:
        print(f"최종 FID 점수 (에폭 {epoch_setting}): {fid_score:.2f}")


def train_gan(epoch_setting):
    print(f"\n{'=' * 50}")
    print(f"에폭 {epoch_setting}으로 학습 시작")
    print(f"{'=' * 50}\n")

    # 코드 실행결과의 동일성을 위해 무작위 시드를 설정합니다
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)

    # 데이터셋의 경로
    dataroot = "../data/lsun_bedroom/data0/lsun/bedroom"

    # dataloader에서 사용할 쓰레드 수
    workers = 4  # 2에서 4로 증가시켜 데이터 로딩 성능 향상

    # 배치 크기
    batch_size = 128

    # 이미지의 크기입니다. 모든 이미지를 변환하여 64로 크기가 통일됩니다.
    image_size = 64

    # 이미지의 채널 수로, RGB 이미지이기 때문에 3으로 설정합니다.
    nc = 3

    # 잠재공간 벡터의 크기 (예. 생성자의 입력값 크기)
    nz = 100

    # 생성자를 통과하는 특징 데이터들의 채널 크기
    ngf = 64

    # 구분자를 통과하는 특징 데이터들의 채널 크기
    ndf = 64

    # 학습할 에폭 수 (인자로 받은 설정 사용)
    num_epochs = epoch_setting

    # TTUR을 위한 서로 다른 학습률 설정
    lr_D = 0.0004  # 판별자 학습률 (더 높게 설정)
    lr_G = 0.0002  # 생성자 학습률 (더 낮게 설정)

    # Adam 옵티마이저의 beta1 하이퍼파라미터
    beta1 = 0.5

    # 사용가능한 gpu 번호. CPU를 사용해야 하는경우 0으로 설정하세요
    ngpu = 1

    # 다양성 손실의 초기 계수 (적응적 가중치 조정에 사용)
    diversity_lambda = 1.0  # 초기값 더 낮게 설정


    # 우리가 설정한 대로 이미지 데이터셋을 불러와 봅시다
    # 먼저 데이터셋을 만듭니다
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    # dataloader를 정의해봅시다
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # GPU 사용여부를 결정해 줍니다
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # ``netG`` 와 ``netD`` 에 적용시킬 커스텀 가중치 초기화 함수
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # 생성된 배치 내 이미지들이 서로 다른 특징들을 가지도록 유도
    def diversity_loss_cosine(fake_features):
        bs = fake_features.shape[0]

        # Flatten하여 벡터화
        reshaped = fake_features.view(bs, -1)  # (batch_size, feature_dim)

        # L2 정규화
        normed = F.normalize(reshaped, p=2, dim=1)

        # 코사인 유사도 계산
        cosine_sim = torch.mm(normed, normed.T)

        # 대각선 제외
        cosine_sim.fill_diagonal_(0)

        # 가중치된 다양성 손실 - 높은 유사도에 더 큰 페널티
        weighted_sim = torch.pow(cosine_sim, 2)

        return weighted_sim.sum() / (bs * (bs - 1))

    # 다양성 메트릭 계산 함수 (모니터링용)
    def calculate_diversity_metric(fake_features):
        bs = fake_features.size(0)
        if bs <= 1:
            return 0.0

        # 이미지를 평탄화하고 정규화
        flat_features = fake_features.view(bs, -1)
        norm_features = F.normalize(flat_features, p=2, dim=1)

        # 코사인 유사도 계산
        sim_matrix = torch.mm(norm_features, norm_features.T)
        sim_matrix.fill_diagonal_(0)

        # 평균 유사도 (낮을수록 다양성이 높음)
        avg_sim = sim_matrix.sum() / (bs * (bs - 1))

        return avg_sim.item()

    # 수정된 Generator 클래스 (중간층 특징맵도 반환)
    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu

            # 초기 특징 추출 (잠재 공간 -> 중간 특징)
            self.features_mid = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
            )

            # 중간 특징 -> 최종 특징
            self.features_final = nn.Sequential(
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
            )

            # 최종 출력 레이어
            self.output = nn.Sequential(
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, input):
            # 중간 특징맵 추출
            mid_feat = self.features_mid(input)

            # 최종 특징맵 추출
            final_feat = self.features_final(mid_feat)

            # 이미지 생성
            img = self.output(final_feat)

            # 이미지와 두 개의 특징맵 반환
            return img, final_feat, mid_feat

    # 수정된 판별자 (스펙트럴 정규화 추가 및 특징 추출 기능)
    from torch.nn.utils import spectral_norm

    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu

            # 초기 특징 추출
            self.features_init = nn.Sequential(
                spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True)
            )

            # 중간 특징 추출
            self.features_mid = nn.Sequential(
                spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True)
            )

            # 최종 특징 추출
            self.features_final = nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True)
            )

            # 판별값 출력
            self.output = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input, return_features=False):
            init_feat = self.features_init(input)
            mid_feat = self.features_mid(init_feat)
            final_feat = self.features_final(mid_feat)
            out = self.output(final_feat)

            if return_features:
                return out.view(-1), final_feat,mid_feat
            return out.view(-1)

    # 생성자를 만듭니다
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    print(netG)

    # 구분자를 만듭니다
    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    print(netD)

    # ``BCELoss`` 함수의 인스턴스를 초기화합니다
    criterion = nn.BCELoss()

    # 학습에 사용되는 참/거짓의 라벨을 정합니다
    real_label = 1.
    fake_label = 0.

    # G와 D에서 TTUR 학습률을 적용한 Adam 옵티마이저를 생성합니다
    optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))

    # 학습률 스케줄러 추가 (성능 안정화를 위해)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=num_epochs, eta_min=lr_D * 0.1)
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=num_epochs, eta_min=lr_G * 0.1)

    # 다양성 지표 추적 및 저장 주기 설정
    diversity_scores = []
    d_losses = []
    g_losses = []
    lambda_values = []  # 다양성 람다 값 추적을 위한 배열 추가
    nash_deviations = []  # 내쉬 균형 편차 추적을 위한 배열 추가

    # 내쉬 균형 기반 다양성 조정을 위한 설정
    original_diversity_lambda = diversity_lambda  # 초기 다양성 가중치 저장
    beta = 5.0  # 내쉬 균형 편차 민감도 계수 (β)
    max_penalty = 10.0  # 최대 페널티 제한

    # 학습 과정
    print("Starting Training Loop...")
    # 에폭(epoch) 반복
    for epoch in range(num_epochs):

        # 한 에폭 내에서 배치 반복
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) D 신경망을 업데이트 합니다
            ###########################
            ## 진짜 데이터들로 학습을 합니다
            netD.zero_grad()
            # 배치들의 사이즈나 사용할 디바이스에 맞게 조정합니다
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # 진짜 데이터들로 이루어진 배치를 D에 통과시킵니다
            output, real_features, d_real_mid_features = netD(real_cpu, return_features=True)
            # 손실값을 구합니다
            errD_real = criterion(output, label)
            # 역전파의 과정에서 변화도를 계산합니다
            errD_real.backward()
            D_x = output.mean().item()

            ## 가짜 데이터들로 학습을 합니다
            # 생성자에 사용할 잠재공간 벡터를 생성합니다
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # G를 이용해 가짜 이미지를 생성합니다
            fake_img, fake_final_feat, fake_mid_feat = netG(noise)
            label.fill_(fake_label)
            # D를 이용해 데이터의 진위를 판별합니다
            output, fake_features, d_fake_mid_features = netD(fake_img.detach(), return_features=True)
            # D의 손실값을 계산합니다
            errD_fake = criterion(output, label)
            # 역전파를 통해 변화도를 계산합니다
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # 가짜 이미지와 진짜 이미지 모두에서 구한 손실값들을 더합니다 + diversity_loss
            diversity_loss_d_real_mid = diversity_loss_cosine(d_real_mid_features)
            diversity_loss_d_fake_mid = diversity_loss_cosine(d_fake_mid_features)

            # 내쉬 균형 편차 계산 (D(x)와 D(G(z))가 0.5에서 얼마나 벗어났는지)
            nash_deviation = abs(D_x - 0.5) + abs(D_G_z1 - 0.5)

            # 편차에 따른 다양성 가중치 조정
            import math
            effective_lambda = original_diversity_lambda * math.exp(beta * nash_deviation)

            # 가중치 제한 (수치적 안정성을 위해)
            min_lambda = original_diversity_lambda / 5.0
            max_lambda = original_diversity_lambda * 5.0
            effective_lambda = min(max(effective_lambda, min_lambda), max_lambda)


            epsilon = 1e-8
            weight = diversity_loss_d_fake_mid / (diversity_loss_d_real_mid + epsilon)
            weight = torch.clamp(weight, min=1.0, max=2.0)

            # 조정된 다양성 페널티 적용
            penalty = effective_lambda * weight * (diversity_loss_d_fake_mid + diversity_loss_d_real_mid)

            # 최대 페널티 제한 (안전장치)
            penalty = min(float(penalty), max_penalty)

            errD = errD_real + errD_fake + penalty
            # D를 업데이트 합니다
            optimizerD.step()

            ############################
            # (2) G 신경망을 업데이트 합니다
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # 생성자의 손실값을 구하기 위해 진짜 라벨을 이용할 겁니다
            # 우리는 방금 D를 업데이트했기 때문에, D에 다시 가짜 데이터를 통과시킵니다.
            output, fake_features,_  = netD(fake_img, return_features=True)

            # 기본 GAN 손실
            basic_g_loss = criterion(output, label)

            # G의 손실값을 구합니다
            errG = basic_g_loss

            # G의 변화도를 계산합니다
            errG.backward()
            D_G_z2 = output.mean().item()

            # G를 업데이트 합니다
            optimizerG.step()

            # 다양성 메트릭 계산 (모니터링용)
            if i % 50 == 0:
                with torch.no_grad():
                    div_score = calculate_diversity_metric(fake_final_feat)
                    diversity_scores.append(div_score)
                    d_losses.append(errD.item())
                    g_losses.append(errG.item())
                    lambda_values.append(effective_lambda)  # 다양성 람다 값 기록
                    nash_deviations.append(nash_deviation)  # 내쉬 균형 편차 기록

                # 훈련 상태를 출력합니다
                print(
                    '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tDiv: %.4f\tNash: %.4f\tλ: %.2f'
                    % (epoch, num_epochs, i, len(dataloader),
                       errD.item(), errG.item(), D_x, D_G_z1, D_G_z2,
                       div_score, nash_deviation, effective_lambda))

        # 에폭 종료 후 학습률 스케줄러 업데이트
        schedulerD.step()
        schedulerG.step()

    # 다양성 메트릭 추이 시각화
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.plot(diversity_scores)
    plt.title(f'Diversity Score Evolution (Epoch {epoch_setting})')
    plt.xlabel('Iterations (x50)')
    plt.ylabel('Avg. Cosine Similarity (Lower is better)')

    plt.subplot(1, 4, 2)
    plt.plot(d_losses, label='D Loss')
    plt.plot(g_losses, label='G Loss')
    plt.title(f'Training Losses (Epoch {epoch_setting})')
    plt.xlabel('Iterations (x50)')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(lambda_values)
    plt.title(f'Diversity Lambda Evolution (Epoch {epoch_setting})')
    plt.xlabel('Iterations (x50)')
    plt.ylabel('Lambda Value')

    plt.subplot(1, 4, 4)
    plt.plot(nash_deviations)
    plt.title(f'Nash Equilibrium Deviation (Epoch {epoch_setting})')
    plt.xlabel('Iterations (x50)')
    plt.ylabel('|D(x) - 0.5| + |D(G(z)) - 0.5|')

    plt.tight_layout()
    plt.savefig(f'training_metrics_epoch_{epoch_setting}.png')

    # 최종 FID 측정을 위한 이미지 생성
    # 저장할 디렉토리 설정 (에폭 설정에 따라 다른 경로 사용)
    real_dir = f"../CosDivGAN_epoch_{epoch_setting}_real_images"
    fake_dir = f"../CosDivGAN_epoch_{epoch_setting}_fake_images"
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # 1. 50,000장 샘플링하여 저장
    num_samples = 50000
    real_samples = []
    fake_samples = []

    # 배치 단위로 이미지 저장 (전체 배치 반복)
    num_full_batches = num_samples // batch_size
    remaining_samples = num_samples % batch_size

    with torch.no_grad():
        data_iter = iter(dataloader)
        for i in range(num_full_batches):
            try:
                real_batch = next(data_iter)[0].to(device)
            except StopIteration:
                data_iter = iter(dataloader)
                real_batch = next(data_iter)[0].to(device)

            real_samples.append(real_batch)
            fake_img, _, _ = netG(torch.randn(batch_size, nz, 1, 1, device=device))
            fake_samples.append(fake_img)

        # 마지막 남은 이미지 처리
        if remaining_samples > 0:
            try:
                real_batch = next(data_iter)[0][:remaining_samples].to(device)
            except StopIteration:
                data_iter = iter(dataloader)
                real_batch = next(data_iter)[0][:remaining_samples].to(device)

            real_samples.append(real_batch)
            fake_img, _, _ = netG(torch.randn(remaining_samples, nz, 1, 1, device=device))
            fake_samples.append(fake_img)

    real_images = torch.cat(real_samples, dim=0)
    fake_images = torch.cat(fake_samples, dim=0)

    # 실제 이미지 저장
    for i in range(real_images.shape[0]):
        vutils.save_image(real_images[i], f"{real_dir}/img_{i}.png", normalize=True)

    # 생성된 이미지 저장
    for i in range(fake_images.shape[0]):
        vutils.save_image(fake_images[i], f"{fake_dir}/img_{i}.png", normalize=True)

    # 최종 FID 계산
    fid_score = fid.compute_fid(real_dir, fake_dir, mode="clean", dataset_res=64, num_workers=0)
    print(f"Final FID Score (에폭 {epoch_setting}, 50,000 images, 64x64 resolution): {fid_score:.2f}")

    # 최종 모델 저장 (에폭 설정 정보 포함)
    save_final_model(netG, netD, fid_score, epoch_setting)

    return fid_score


def main():
    # 여러 에폭 설정으로 학습 실행
    epoch_settings = [20,30,45,60,75,90]

    # 각 에폭 설정에 대한 결과 저장
    results = {}

    # 각 에폭 설정으로 학습 실행
    for epoch_setting in epoch_settings:
        print(f"\n{'#' * 80}")
        print(f"# 에폭 {epoch_setting}으로 학습 시작")
        print(f"{'#' * 80}\n")

        # 지정된 에폭으로 학습 실행
        fid_score = train_gan(epoch_setting)

        # 결과 저장
        results[epoch_setting] = fid_score

    # 학습 결과 요약
    print("\n" + "=" * 50)
    print("학습 결과 요약")
    print("=" * 50)

    for epoch, score in results.items():
        print(f"에폭 {epoch}: FID 점수 {score:.2f}")


if __name__ == '__main__':
    # CuBLAS 결정적 알고리즘 설정 (CUDA 10.2 이상 필수)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 결정적 알고리즘 활성화
    torch.use_deterministic_algorithms(True)

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()