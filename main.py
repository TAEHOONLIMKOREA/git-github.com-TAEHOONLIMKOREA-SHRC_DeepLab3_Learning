from Module.TrainMethods import data_generator, read_image, train, continue_train
from Module.InferMethods import plot_random_predictions, calculate_accuracy_from_dataset, generate_synthetic_accuracy, calculate_random_accuracy, create_colormap, infer
from Module.ProcessingMethods import display_images_and_masks, validate_count_of_images_and_masks, validate_mask_classes
from tensorflow import keras
from keras import layers
import os
from glob import glob


NUM_CLASSES = 6
# DATA_DIR = "./SHRC_GarbageDetection/Data/TrainDataSet"
DATA_DIR = "/home/keti_taehoon/SHRC_DeepLab3Plus_Learning/Data/TrainDataSet_Oct_50m"
# LBAEL_PATH = "./SHRC_GarbageDetection/Data/TrainDataSet/labelmap.txt"
LBAEL_PATH = "/home/keti_taehoon/SHRC_DeepLab3Plus_Learning/Data/TrainDataSet_Oct_50m/labelmap.txt"
# NUM_TRAIN_IMAGES = 669
# NUM_VAL_IMAGES = 50

NUM_TRAIN_IMAGES = 2500
NUM_VAL_IMAGES = 485



def main():    
    # [1] 훈련용 데이터셋 준비
    train_images = sorted(glob(os.path.join(DATA_DIR, 'Aug_Images/*')))[:NUM_TRAIN_IMAGES]
    train_masks = sorted(glob(os.path.join(DATA_DIR, 'Aug_SegmentationClass/*')))[:NUM_TRAIN_IMAGES]
    val_images = sorted(glob(os.path.join(DATA_DIR, 'Aug_Images/*')))[NUM_TRAIN_IMAGES:NUM_VAL_IMAGES+NUM_TRAIN_IMAGES]
    val_masks = sorted(glob(os.path.join(DATA_DIR, 'Aug_SegmentationClass/*')))[NUM_TRAIN_IMAGES:NUM_VAL_IMAGES+NUM_TRAIN_IMAGES]
    
    print(len(train_images))
    print(len(train_masks))
    # validate_count_of_images_and_masks(train_images)
    # print(os.getcwd())
    train_dataset = data_generator(train_images, train_masks, LBAEL_PATH)
    val_dataset = data_generator(val_images, val_masks, LBAEL_PATH)


    
    # 출력된 클래스 ID가 0 ~ NUM_CLASSES - 1 범위인지 확인하는 함수
    # validate_mask_classes(val_dataset)

    # 배치 사이즈 만큼 무작위로 사진과 마스크 출력
    # display_images_and_masks(train_dataset)
    
    # [1-1] 학습시
    # model = train(train_dataset, val_dataset)
    # [1-2]추가 학습시
    model_path = '/home/keti_taehoon/SHRC_DeepLab3Plus_Learning/saved_model/continued_model_2025-11-17_17-43-53.keras'
    continue_train(model_path, train_dataset, val_dataset, epochs=20)
        
    # [2] 모델 불러오기
    # model = keras.models.load_model('SHRC_GarbageDetection/saved_model/continued_model_2025-11-04_12-15-16.keras')
    # model = layers.TFSMLayer(
    #     "SHRC_GarbageDetection/saved_model/model_2025-11-04 07:41:14",
    #     call_endpoint="serving_default"  # 대부분 이 이름을 씁니다
    # )
    

    # [3] 컬러맵 생성 
    # colormap = create_colormap(LBAEL_PATH)    
    
    # [4] 추론
    # plot_random_predictions(train_images, colormap, model=model)    
    # acc = calculate_accuracy_from_dataset(val_dataset, model, "Random_Test_Acc")
    # acc = calculate_random_accuracy(train_dataset, model, "Random_Test_Acc")
    # generate_synthetic_accuracy()
    # print(acc)

if __name__ == '__main__':
    main()