from Module.TrainMethods import data_generator, read_image, train
from Module.GarbageDetectMethods import plot_random_predictions, calculate_accuracy_from_dataset, generate_synthetic_accuracy, calculate_random_accuracy
from Module.InferMethods import create_colormap
from Module.ProcessingMethods import display_images_and_masks, validate_count_of_images_and_masks, validate_mask_classes
from tensorflow import keras
import os
from glob import glob


NUM_CLASSES = 6
DATA_DIR = "./SHRC_GarbageDetection/Data/TrainDataSet"
NUM_TRAIN_IMAGES = 669
NUM_VAL_IMAGES = 50


def main():    
    # [1] 훈련용 데이터셋 준비
    train_images = sorted(glob(os.path.join(DATA_DIR, 'Images/*')))[:NUM_TRAIN_IMAGES]
    train_masks = sorted(glob(os.path.join(DATA_DIR, 'SegmentationClass/*')))[:NUM_TRAIN_IMAGES]
    val_images = sorted(glob(os.path.join(DATA_DIR, 'Images/*')))[NUM_TRAIN_IMAGES:NUM_VAL_IMAGES+NUM_TRAIN_IMAGES]
    val_masks = sorted(glob(os.path.join(DATA_DIR, 'SegmentationClass/*')))[NUM_TRAIN_IMAGES:NUM_VAL_IMAGES+NUM_TRAIN_IMAGES]
    
    print(len(train_images))
    print(len(train_masks))
    # validate_count_of_images_and_masks(train_images)
    # print(os.getcwd())
    label_path = "./SHRC_GarbageDetection/Data/TrainDataSet/labelmap.txt"
    train_dataset = data_generator(train_images, train_masks, label_path)
    val_dataset = data_generator(val_images, val_masks, label_path)


    
    # 출력된 클래스 ID가 0 ~ NUM_CLASSES - 1 범위인지 확인하는 함수
    # validate_mask_classes(val_dataset)

    # 배치 사이즈 만큼 무작위로 사진과 마스크 출력
    # display_images_and_masks(train_dataset)
    
    # [1] 학습  
    # model = train(train_dataset, val_dataset)
    
    # [2] 모델 불러오기
    model = keras.models.load_model('SHRC_GarbageDetection/model_2025-01-21 16:53:49.h5')

    # [3] 컬러맵 생성 
    colormap = create_colormap(label_path)    
    
    # [4] 추론
    plot_random_predictions(train_images, colormap, model=model)    
    acc = calculate_accuracy_from_dataset(val_dataset, model, "Random_Test_Acc")
    acc = calculate_random_accuracy(train_dataset, model, "Random_Test_Acc")
    # generate_synthetic_accuracy()
    # print(acc)

if __name__ == '__main__':
    main()