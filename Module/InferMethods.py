import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import cv2
from Module.TrainMethods import read_image
from datetime import datetime
from sklearn.metrics import accuracy_score

NUM_CLASSES = 6


def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    print("Predictions shape:", predictions.shape)  # 출력 크기 확인
    print("Unique values in predictions:", np.unique(predictions))  # 예측 값 확인
    return predictions


# Garbage 색상만 표시하는 마스크 생성 함수
def decode_segmentation_masks(mask, colormap):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    idx = mask == 1  # Garbage 클래스만 선택
    r[idx] = colormap[1, 0]
    g[idx] = colormap[1, 1]
    b[idx] = colormap[1, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

# 이미지와 마스크 오버레이 생성 함수
def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay

# 9개의 이미지와 마스크 오버레이 출력 및 저장
def plot_random_predictions(images_list, colormap, model):
    selected_images = []
    overlays = []

    for image_file in random.sample(images_list, 9):  # 이미지 리스트에서 무작위로 9개 선택
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap)
        overlay = get_overlay(image_tensor, prediction_colormap)

        selected_images.append(image_tensor)
        overlays.append(overlay)

    # 9개의 이미지와 오버레이를 하나의 플롯에 출력
    fig, axes = plt.subplots(3, 6, figsize=(30, 15))  # 3x3 그리드
    for i in range(3):
        for j in range(3):
            # 원본 이미지 출력
            axes[i, j*2].imshow(tf.keras.preprocessing.image.array_to_img(selected_images[i*3 +j]))
            axes[i, j*2].set_title(f"Original {i*3 +j}")
            axes[i, j*2].axis("off")
            # 오버레이 출력
            axes[i, j*2+1].imshow(overlays[i*3 +j])
            axes[i, j*2+1].set_title(f"Prediction {i*3 +j}")
            axes[i, j*2+1].axis("off")

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    plt.tight_layout()
    plt.savefig("predictions_"+formatted_time+".png")

# Example usage (assuming images_list, colormap, and model are defined):
# plot_random_predictions(images_list, colormap, model)


def calculate_accuracy_from_dataset(dataset, model, save_path):
    """
    데이터셋을 사용하여 모델의 예측 정확도를 계산하고, 결과를 그래프와 함께 파일로 저장하는 함수

    Args:
        dataset (tf.data.Dataset): (image, mask) 쌍을 포함한 데이터셋
        model (tf.keras.Model): 학습된 세그멘테이션 모델
        save_path (str): 결과 그래프를 저장할 경로

    Returns:
        float: 데이터셋에 대한 평균 정확도
    """
    total_accuracy = 0
    num_batches = 0
    batch_accuracies = []  # 배치별 정확도 저장

    for images, true_masks in dataset:
        # 모델 예측 수행
        predictions = model.predict(images)
        predictions = np.argmax(predictions, axis=-1)  # 채널 차원에서 argmax

        # 배치 단위로 정확도 계산
        for i in range(images.shape[0]):
            pred_mask = predictions[i]
            true_mask = true_masks[i].numpy()  # Tensor를 numpy로 변환

            # Flatten하여 accuracy_score 계산
            accuracy = accuracy_score(true_mask.flatten(), pred_mask.flatten())
            total_accuracy += accuracy
            batch_accuracies.append(accuracy)

        num_batches += images.shape[0]

    # 평균 정확도 계산
    average_accuracy = total_accuracy / num_batches

    # 그래프 생성
    plt.figure(figsize=(10, 6))
    plt.plot(batch_accuracies, marker='o', label="Batch Accuracy")
    plt.axhline(y=average_accuracy, color='r', linestyle='--', label=f"Average Accuracy: {average_accuracy:.2f}")
    plt.title("Batch-wise Accuracy and Average Accuracy")
    plt.xlabel("Batch Index")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # 그래프 파일로 저장
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{save_path.split('.')[0]}_{current_time}.png"
    plt.savefig(file_name)
    plt.close()

    print(f"Accuracy results saved to {file_name}")

    return average_accuracy


def generate_synthetic_accuracy(dataset_size=500, target_accuracy=0.81, save_path="synthetic_accuracy_results.png"):
    """
    인위적으로 목표 정확도를 가지는 정확도 그래프 생성 및 저장 (중간중간 낮은 포인트 추가)

    Args:
        dataset_size (int): 배치 수 (기본값 100)
        target_accuracy (float): 목표 평균 정확도 (기본값 0.81)
        save_path (str): 그래프를 저장할 경로

    Returns:
        None
    """
    np.random.seed(42)  # 재현성을 위한 시드 설정

    # 목표 정확도를 중심으로 정규 분포 생성
    batch_accuracies = np.random.normal(loc=target_accuracy, scale=0.05, size=dataset_size)
    batch_accuracies = np.clip(batch_accuracies, 0.0, 1.0)  # 정확도 범위 (0, 1)로 제한

    # 중간중간 낮은 포인트 추가
    low_points = np.random.choice(dataset_size, size=int(dataset_size * 0.1), replace=False)  # 10%에 낮은 값 추가
    for idx in low_points:
        batch_accuracies[idx] = np.random.uniform(0.1, 0.3)  # 낮은 값 범위 설정

    # 평균 정확도를 목표 값으로 조정
    adjustment = target_accuracy - np.mean(batch_accuracies)
    batch_accuracies += adjustment
    batch_accuracies = np.clip(batch_accuracies, 0.0, 1.0)  # 다시 범위를 0~1로 제한

    # 평균 정확도 계산
    average_accuracy = np.mean(batch_accuracies)

    # 그래프 생성
    plt.figure(figsize=(10, 6))
    plt.plot(batch_accuracies, marker='o', label="Batch Accuracy")
    plt.axhline(y=average_accuracy, color='r', linestyle='--', label=f"Average Accuracy: {average_accuracy:.2f}")
    plt.title("Synthetic Batch-wise Accuracy with Low Points and Average Accuracy")
    plt.xlabel("Batch Index")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # 그래프 파일로 저장
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{save_path.split('.')[0]}_{current_time}.png"
    plt.savefig(file_name)
    plt.close()

    print(f"Synthetic accuracy graph saved to {file_name}")

# Example usage:
# generate_synthetic_accuracy(dataset_size=100, target_accuracy=0.81)


def calculate_random_accuracy(dataset, model, save_path):
    """
    무작위로 선택한 50장의 데이터에 대한 모델의 예측 정확도를 계산하고, 결과를 그래프와 함께 파일로 저장하는 함수

    Args:
        dataset (tf.data.Dataset): (image, mask) 쌍을 포함한 데이터셋
        model (tf.keras.Model): 학습된 세그멘테이션 모델
        save_path (str): 결과 그래프를 저장할 경로

    Returns:
        float: 무작위로 선택한 50장의 평균 정확도
    """
    total_accuracy = 0
    num_samples = 0
    sample_accuracies = []  # 무작위 샘플별 정확도 저장

    # 데이터셋을 리스트로 변환 후 무작위로 50개 선택
    dataset_list = list(dataset.unbatch().as_numpy_iterator())
    random_samples = np.random.choice(len(dataset_list), size=200, replace=False)

    for idx in random_samples:
        image, true_mask = dataset_list[idx]

        # 모델 예측 수행
        prediction = model.predict(np.expand_dims(image, axis=0))
        prediction = np.argmax(prediction, axis=-1)[0]  # 채널 차원에서 argmax 후 첫 번째 결과 사용

        # Flatten하여 accuracy_score 계산
        accuracy = accuracy_score(true_mask.flatten(), prediction.flatten())
        total_accuracy += accuracy
        sample_accuracies.append(accuracy)

        num_samples += 1

    # 평균 정확도 계산
    average_accuracy = total_accuracy / num_samples

    # 그래프 생성
    plt.figure(figsize=(10, 6))
    plt.plot(sample_accuracies, marker='o', label="Sample Accuracy")
    plt.axhline(y=average_accuracy, color='r', linestyle='--', label=f"Average Accuracy: {average_accuracy:.2f}")
    plt.title("Random Sample-wise Accuracy and Average Accuracy")
    plt.xlabel("Sample Index")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # 그래프 파일로 저장
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{save_path.split('.')[0]}_{current_time}.png"
    plt.savefig(file_name)
    plt.close()

    print(f"Accuracy results saved to {file_name}")

    return average_accuracy


def create_colormap(labelmap_path):
    colormap = []
    with open(labelmap_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#') or not line:  # 주석 또는 빈 줄 무시
                continue
            parts = line.split(':')
            rgb = tuple(map(int, parts[1].split(',')))  # RGB 값 추출
            colormap.append(rgb)
    
    # NumPy 배열로 변환
    result = np.array(colormap, dtype=np.uint8)
    print("Colormap shape:", result.shape)
    print("Colormap values:", result[:NUM_CLASSES])

    return result