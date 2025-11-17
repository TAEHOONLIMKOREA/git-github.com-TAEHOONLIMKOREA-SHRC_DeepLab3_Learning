from glob import glob
from scipy.io import loadmat
import json
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

IMAGE_SIZE = 512
BATCH_SIZE = 16
NUM_CLASSES = 20

def parse_labelmap(labelmap_path):
    class_to_rgb = {}
    with open(labelmap_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#') or not line:  # 주석 또는 빈 줄 무시
                continue
            parts = line.split(':')
            label = parts[0]  # 클래스 이름
            rgb = tuple(map(int, parts[1].split(',')))  # RGB 값 튜플로 변환
            class_to_rgb[label] = rgb
    print(class_to_rgb)
    return class_to_rgb

def convert_mask_to_labels(mask, class_to_rgb):
    # RGB -> 클래스 ID 매핑
    rgb_to_class = {rgb: idx for idx, (cls, rgb) in enumerate(class_to_rgb.items())}
    mask_shape = mask.shape[:2]  # 높이와 너비
    label_mask = tf.zeros(mask_shape, dtype=tf.int32)  # label_mask를 int32로 초기화

    for rgb, class_id in rgb_to_class.items():
        # RGB 값을 기준으로 마스크의 픽셀을 클래스 ID로 매핑
        condition = tf.reduce_all(mask == rgb, axis=-1)  # 동일한 RGB 값 찾기
        label_mask = tf.where(condition, tf.cast(class_id, tf.int32), label_mask)  # 타입 일치
    return label_mask

def read_image(image_path, mask=False, class_to_rgb=None):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        # 세그멘테이션 마스크는 Nearest Neighbor로 리사이즈 권장(범주형 레이블 보존)
        image = tf.image.resize(
            images=image,
            size=[IMAGE_SIZE, IMAGE_SIZE],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        # 필요하면 정수형 캐스팅
        image = tf.cast(image, tf.uint8)
        if class_to_rgb:
            image = convert_mask_to_labels(image, class_to_rgb)  # 레이블 변환
    else:
        image = tf.image.decode_jpeg(image, channels=3)
        image.set_shape([None, None, 3])  # RGB
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        # -1 ~ 1 범위로 정규화
        image = image / 127.5 - 1.0
    return image

def load_data(image_list, mask_list, class_to_rgb):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True, class_to_rgb=class_to_rgb)
    return image, mask

def data_generator(image_list, mask_list, labelmap_path):
    class_to_rgb = parse_labelmap(labelmap_path)
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(lambda img, mask: load_data(img, mask, class_to_rgb), 
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, 
                      padding='same', use_bias=False):
    x = layers.Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=HeNormal()
    )(block_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    # Global Average Pooling branch
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(size=(dims[-3] // x.shape[1], dims[-2]), 
                                    interpolation='bilinear')(x)

    # Dilated Convs
    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3(num_classes):
    # 입력 정의
    model_input = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    
    # ResNet50 백본
    resnet50 = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=model_input
    )
    
    # conv4_block6_2_relu 출력
    x = resnet50.get_layer('conv4_block6_2_relu').output
    x = DilatedSpatialPyramidPooling(x)
    
    # Upsampling 1
    input_a = layers.UpSampling2D(
        size=(IMAGE_SIZE // 4 // x.shape[1], IMAGE_SIZE // 4 // x.shape[2]),
        interpolation='bilinear'
    )(x)
    
    # 저수준 특징: conv2_block3_2_relu
    input_b = resnet50.get_layer('conv2_block3_2_relu').output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)
    
    # Feature Fusion
    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    
    # 최종 업샘플
    x = layers.UpSampling2D(
        size=(IMAGE_SIZE // x.shape[1], IMAGE_SIZE // x.shape[2]),
        interpolation='bilinear'
    )(x)
    
    # 클래스 채널
    model_output = layers.Conv2D(num_classes, kernel_size=(1,1), padding='same')(x)
    
    return tf.keras.Model(inputs=model_input, outputs=model_output)

def train(image_list, mask_list, labelmap_path, k_folds=5, epochs=30, learning_rate=1e-5):
    """
    K-Fold Cross Validation으로 자동 학습
    
    Args:
        image_list: 이미지 파일 경로 리스트
        mask_list: 마스크 파일 경로 리스트
        labelmap_path: 레이블맵 파일 경로
        k_folds: Fold 개수 (기본값: 5)
        epochs: 각 fold당 에폭 수 (기본값: 30)
        learning_rate: 학습률 (기본값: 1e-5)
    
    Returns:
        최고 성능 모델
    """
    # [1] GPU 메모리 설정
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # [2] 데이터를 numpy 배열로 변환
    image_array = np.array(image_list)
    mask_array = np.array(mask_list)
    
    # [3] K-Fold 설정
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # [4] MirroredStrategy 초기화
    strategy = tf.distribute.MirroredStrategy()
    
    best_model = None
    best_val_loss = float('inf')
    all_histories = []
    
    # [5] K-Fold 학습
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(image_array)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{k_folds} 학습 중...")
        print(f"{'='*60}")
        
        # Train/Validation 데이터 분리
        train_images = image_array[train_idx].tolist()
        train_masks = mask_array[train_idx].tolist()
        val_images = image_array[val_idx].tolist()
        val_masks = mask_array[val_idx].tolist()
        
        # 데이터셋 생성
        train_dataset = data_generator(train_images, train_masks, labelmap_path)
        val_dataset = data_generator(val_images, val_masks, labelmap_path)
        
        # 모델 생성 및 학습
        with strategy.scope():
            model = DeeplabV3(num_classes=NUM_CLASSES)
            if fold_idx == 0:
                model.summary()
            
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss=loss,
                metrics=['accuracy']
            )
        
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
        all_histories.append(history.history)
        
        # 최고 성능 모델 저장
        final_val_loss = history.history['val_loss'][-1]
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_model = model
            print(f"✓ 최고 성능 모델 갱신! (Val Loss: {final_val_loss:.4f})")
        
        # 각 fold 모델 저장
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model.save(f'SHRC_GarbageDetection/saved_model/fold_{fold_idx+1}_{current_time}.keras')
    
    # [6] 결과 출력 및 저장
    print(f"\n{'='*60}")
    print("K-Fold 학습 완료!")
    print(f"{'='*60}")
    
    # 평균 성능 계산
    final_train_loss = [h['loss'][-1] for h in all_histories]
    final_val_loss = [h['val_loss'][-1] for h in all_histories]
    final_train_acc = [h['accuracy'][-1] for h in all_histories]
    final_val_acc = [h['val_accuracy'][-1] for h in all_histories]
    
    print(f"\n평균 성능:")
    print(f"  Train Loss: {np.mean(final_train_loss):.4f} (±{np.std(final_train_loss):.4f})")
    print(f"  Val Loss:   {np.mean(final_val_loss):.4f} (±{np.std(final_val_loss):.4f})")
    print(f"  Train Acc:  {np.mean(final_train_acc):.4f} (±{np.std(final_train_acc):.4f})")
    print(f"  Val Acc:    {np.mean(final_val_acc):.4f} (±{np.std(final_val_acc):.4f})")
    
    # [7] 최고 모델 저장
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    best_model_path = f'SHRC_GarbageDetection/saved_model/best_model_{current_time}.keras'
    best_model.save(best_model_path)
    print(f"\n최고 성능 모델 저장: {best_model_path}")
    
    # [8] History 저장
    with open(f'kfold_history_{current_time}.json', 'w') as f:
        json.dump(all_histories, f)
    
    # [9] 시각화
    _plot_results(all_histories, current_time)
    
    return best_model

def _plot_results(all_histories, timestamp):
    """학습 결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Train Loss
    for i, history in enumerate(all_histories):
        axes[0, 0].plot(history['loss'], label=f'Fold {i+1}')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Validation Loss
    for i, history in enumerate(all_histories):
        axes[0, 1].plot(history['val_loss'], label=f'Fold {i+1}')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Train Accuracy
    for i, history in enumerate(all_histories):
        axes[1, 0].plot(history['accuracy'], label=f'Fold {i+1}')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Validation Accuracy
    for i, history in enumerate(all_histories):
        axes[1, 1].plot(history['val_accuracy'], label=f'Fold {i+1}')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'kfold_results_{timestamp}.png', dpi=300)
    print(f"결과 그래프 저장: kfold_results_{timestamp}.png")

def continue_train(model_path, train_dataset, val_dataset, epochs=10, learning_rate=1e-5):
    """
    기존과 동일하게 사용 가능한 continue_train 함수
    """
    # [1] GPU 메모리 설정
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # [2] MirroredStrategy 초기화
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # [3] 기존 모델 로드
        print(f"모델 로드 중: {model_path}")
        model = keras.models.load_model(model_path)
        model.summary()
        print("Model output shape:", model.output_shape)

        # [4] 모델 재컴파일
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=['accuracy']
        )

    # [5] 추가 학습 진행
    print(f"\n추가 학습 시작 ({epochs} 에포크)")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
    )

    # [6] 모델 저장
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    model.save(f'SHRC_GarbageDetection/saved_model/continued_model_{formatted_time}.keras')
    save_path = f'SHRC_GarbageDetection/saved_model/continued_model_{formatted_time}'
    model.export(save_path)
    print(f"\n모델이 성공적으로 저장되었습니다: {save_path}")

    return history, model