import time
import argparse
import pandas as pd
from src.utils import Logger, Setting, models_load, parse_args, parse_args_boolean
from src.data import context_data_load, context_data_split, context_data_loader
from src.data import dl_data_load, dl_data_split, dl_data_loader, context_dl_data_load
from src.data import image_data_load, image_data_split, image_data_loader
from src.data import text_data_load, text_data_split, text_data_loader
from src.data import ml_data_load, ml_data_split
from src.train import train, test, ml_train, ml_test
from src.ml_config.CatBoost import CatBoostConfig
from src.ml_config.XGBoost import XGBoostConfig
from src.ml_config.LightGBM import LightGBMConfig
import wandb

def main(args):
    Setting.seed_everything(args.seed)

    # text vector를 생성
    if args.vector_create:
        import nltk
        nltk.download('punkt')
        data = text_data_load(args)
        exit()

    ######################## DATA LOAD
    print(f'--------------- {args.model} Load Data ---------------')
    if args.model in ('FM', 'FFM', 'DeepFFM', 'DeepFM'):
        data = context_data_load(args)
    elif args.model in ('NCF', 'WDN', 'DCN'):
        data = dl_data_load(args)
    elif args.model in ('cNCF', 'cNCF-v2', 'cNCF-v3'):
        data = context_dl_data_load(args)
    elif args.model == 'CNN_FM':
        data = image_data_load(args)
    elif args.model == 'DeepCoNN':
        import nltk
        nltk.download('punkt')
        data = text_data_load(args)
    elif args.model in ('CatBoost', 'XGBoost', 'LightGBM'):
        data = ml_data_load(args)
    else:
        pass


    ######################## Train/Valid Split
    print(f'--------------- {args.model} Train/Valid Split ---------------')
    if args.model in ('FM', 'FFM', 'DeepFFM', 'DeepFM'):
        data = context_data_split(args, data)
        data = context_data_loader(args, data)

    elif args.model in ('NCF', 'cNCF', 'cNCF-v2', 'cNCF-v3', 'WDN', 'DCN'):
        data = dl_data_split(args, data)
        data = dl_data_loader(args, data)
        
    elif args.model=='CNN_FM':
        data = image_data_split(args, data)
        data = image_data_loader(args, data)

    elif args.model=='DeepCoNN':
        data = text_data_split(args, data)
        data = text_data_loader(args, data)

    elif args.model in ('CatBoost', 'XGBoost', 'LightGBM'):
        data = ml_data_split(args, data)

    else:
        pass


    ####################### Setting for Log
    setting = Setting()

    log_path = setting.get_log_path(args)
    setting.make_dir(log_path)

    logger = Logger(args, log_path)
    logger.save_args()

        
    filename = setting.get_submit_filename(args)
    
    if args.wandb:   
    
        ############### wandb initialization
        wandb.init(project='ai-tech-level1-1')

        wandb.run.name = filename[9:-4]
        wandb.run.save()

        wandb.config.update(args)

        if args.model in ('CatBoost',):
            wandb.config.update(CatBoostConfig)
        elif args.model == 'XGBoost':
            wandb.config.update(XGBoostConfig)
        elif args.model == 'LightGBM':
            wandb.config.update(LightGBMConfig)
            

    ######################## Model
    print(f'--------------- INIT {args.model} ---------------')
    model = models_load(args,data)


    ######################## TRAIN
    print(f'--------------- {args.model} TRAINING ---------------')
    if args.model in ('CatBoost', 'XGBoost', 'LightGBM'):
        model = ml_train(args, model, data, logger, setting)
    else:
        model = train(args, model, data, logger, setting)


    ######################## INFERENCE
    print(f'--------------- {args.model} PREDICT ---------------')
    if args.model in ('CatBoost', 'XGBoost', 'LightGBM'):
        predicts = ml_test(args, model, data, setting)
    else:
        predicts = test(args, model, data, setting)


    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    submission = pd.read_csv(args.data_path + 'sample_submission.csv')
    
    if args.model in ('FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN', 'CatBoost', 'DeepFFM', 'XGBoost', 'DeepFM', 'LightGBM'):
        submission['rating'] = predicts
    else:
        pass

    submission.to_csv(filename, index=False)
    
    if args.wandb:
        submission_artifact = wandb.Artifact('submission', type='output')
        submission_artifact.add_file(filename, name=filename[9:-4])
        wandb.log_artifact(submission_artifact)
        wandb.finish()


if __name__ == "__main__":

    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument


    ############### BASIC OPTION
    arg('--data_path', type=str, default='data/', help='Data path를 설정할 수 있습니다.')
    arg('--saved_model_path', type=str, default='./saved_models', help='Saved Model path를 설정할 수 있습니다.')
    arg('--model', type=str, choices=['FM', 'FFM', 'NCF', 'cNCF', 'cNCF-v2', 'cNCF-v3', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN', 'CatBoost', 'DeepFFM', 'DeepFM', 'XGBoost', 'LightGBM'],
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--data_shuffle', type=parse_args_boolean, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--test_size', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--seed', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--use_best_model', type=parse_args_boolean, default=True, help='검증 성능이 가장 좋은 모델 사용여부를 설정할 수 있습니다.')
    arg('--wandb', type=parse_args_boolean, default=True, help='WandB 사용 여부를 설정할 수 있습니다.')

    arg('--preprocess_user', type=parse_args, default=('user_id', 'age', 'location',), help='dl-data에서 users의 전처리 대상 칼럼을 지정할 수 있습니다.')
    arg('--preprocess_book', type=parse_args, default=('isbn', 'book_title', 'year_of_publication', 'publisher', 'language', 'category', 'book_author'), help='dl-data에서 books의 전처리 대상 칼럼을 지정할 수 있습니다.')

    arg('--merge_users', type=parse_args_boolean, default=False, help='dl-data에서 users 정보 사용 여부를 설정할 수 있습니다.')
    arg('--merge_books', type=parse_args_boolean, default=False, help='dl-data에서 books 정보 사용 여부를 설정할 수 있습니다.')
    arg('--wdn_rate', type=float, default=0.8, help='WDN에서 Deep component 의 비중을 조정할 수 있습니다.')
    arg('--minus_rate', type=float, default=0.0, help='WDN에서 Minus Deep component 의 영향도를 조정할 수 있습니다.')
    arg('--rating_bias', type=float, default=7.069714, help='WDN에서 유저의 선호여부를 판단하는 기준을 조정할 수 있습니다.')

    ############### TRAINING OPTION
    arg('--batch_size', type=int, default=1024, help='Batch size를 조정할 수 있습니다.')
    arg('--epochs', type=int, default=10, help='Epoch 수를 조정할 수 있습니다.')
    arg('--lr', type=float, default=1e-3, help='Learning Rate를 조정할 수 있습니다.')
    arg('--loss_fn', type=str, default='RMSE', choices=['MSE', 'RMSE'], help='손실 함수를 변경할 수 있습니다.')
    arg('--optimizer', type=str, default='ADAM', choices=['SGD', 'ADAM'], help='최적화 함수를 변경할 수 있습니다.')
    arg('--weight_decay', type=float, default=1e-6, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')


    ############### GPU
    arg('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')


    ############### FM, FFM, NCF, WDN, DCN, DeepFM, DeepFFM Common OPTION
    arg('--embed_dim', type=int, default=16, help='FM, FFM, NCF, WDN, DCN, DeepFM, DeepFFM에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--dropout', type=float, default=0.2, help='NCF, WDN, DCN, DeepFM, DeepFFM에서 Dropout rate를 조정할 수 있습니다.')
    arg('--mlp_dims', type=parse_args, default=(16, 16), help='NCF, WDN, DCN, DeepFM, DeepFFM에서 MLP Network의 차원을 조정할 수 있습니다.')

    ############## context preprocessing
    arg('--isbn_info', type=parse_args_boolean, default=False, help='isbn의 group, publisher, title 정보의 사용 여부를 설정할 수 있습니다.')
    arg('--drop_city_state', type=parse_args_boolean, default=False, help='location의 city, state 정보의 사용 여부를 설정할 수 있습니다.')
    arg('--cut_category', type=parse_args_boolean, default=False, help='category를 빈도 50 이상의 정보만 사용하게 할 지 여부를 설정할 수 있습니다.')
    arg('--category_impute', type=parse_args_boolean, default=False, help='category를 mode로 imputation 할 것인지 여부를 설정할 수 있습니다.')


    ############### DeepFM OPTION
    arg('--activation_fn', type=str, default='relu', choices=['relu', 'tanh'], help='활성화 함수를 변경할 수 있습니다.')
    arg('--use_bn', type=parse_args_boolean, default=False, help='배치 정규화 사용 여부를 설정할 수 있습니다.')

    arg('--merge_summary', type=parse_args_boolean, default=False, help='book summary 사용 여부를 설정할 수 있습니다.')


    ############### FM OPTION
    arg('--age_continuous', type=parse_args_boolean, default=False, help='age를 continuous 사용 여부를 설정할 수 있습니다. FM에서만 동작합니다.')


    ############### DCN
    arg('--num_layers', type=int, default=3, help='에서 Cross Network의 레이어 수를 조정할 수 있습니다.')


    ############### CNN_FM
    arg('--cnn_embed_dim', type=int, default=64, help='CNN_FM에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--cnn_latent_dim', type=int, default=12, help='CNN_FM에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')


    ############### DeepCoNN
    arg('--vector_create', type=parse_args_boolean, default=False, help='DEEP_CONN에서 text vector 생성 여부를 조정할 수 있으며 최초 학습에만 True로 설정하여야합니다.')
    arg('--deepconn_embed_dim', type=int, default=32, help='DEEP_CONN에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--deepconn_latent_dim', type=int, default=10, help='DEEP_CONN에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')
    arg('--conv_1d_out_dim', type=int, default=50, help='DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.')
    arg('--kernel_size', type=int, default=3, help='DEEP_CONN에서 1D conv의 kernel 크기를 조정할 수 있습니다.')
    arg('--word_dim', type=int, default=768, help='DEEP_CONN에서 1D conv의 입력 크기를 조정할 수 있습니다.')
    arg('--out_dim', type=int, default=32, help='DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.')

    args = parser.parse_args()
    
    main(args)
