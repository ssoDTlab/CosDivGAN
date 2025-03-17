import os
from cleanfid import fid


def main():
    real_dir = "../DCGAN_add_cosine_similarity_real_images"
    fake_dir = "../DCGAN_add_cosine_similarity_fake_images"
    # 2️⃣ Clean-FID로 FID 측정 (64x64 크기 그대로 비교)
    fid_score = fid.compute_fid(real_dir, fake_dir, mode="clean", dataset_res=64, num_workers=0)
    print(f"FID Score (50,000 images, 64x64 resolution): {fid_score:.2f}")

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()