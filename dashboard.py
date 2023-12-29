import streamlit as st
import subprocess

def main():
    st.title('Level-1 BookRatingPrediction')
    
    batch_size = st.number_input("Batch Size", min_value=1, value=1024)
    epochs = st.number_input("Epochs", min_value=1, value=10)
    lr = st.number_input("Learning Rate", min_value=0.0, value=1e-3, format="%e")
    loss_fn = st.selectbox("Loss Function", ['MSE', 'RMSE'])
    optimizer = st.selectbox("Optimizer", ['SGD', 'ADAM'])
    weight_decay = st.number_input("Weight Decay", min_value=0.0, value=1e-6, format="%e")
    device = st.selectbox("Device", ['cuda', 'cpu'])
    args = ""
    args += f" --batch_size {batch_size}"
    args += f" --epochs {epochs}"
    args += f" --lr {lr}"
    args += f" --loss_fn {loss_fn}"
    args += f" --optimizer {optimizer}"
    args += f" --weight_decay {weight_decay}"
    args += f" --device {device}"
    
    model_choice = st.selectbox("Select Model", ['FM', 'FFM' , 'DeepFFM', 'DeepFN' ,' NCF' ,'cNCF', 'cNCF-v2', 'cNCF-v3', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN', 'CatBoost', 'XGBoost'])
    args = f" --model {model_choice}"
    
    if model_choice in ('FM', 'FFM', 'NCF', 'cNCF', 'cNCF-v2', 'cNCF-v3', 'WDN', 'DCN', 'DeepFM', 'DeepFFM'):
        embed_dim = st.number_input('Embedding Dimension', min_value=1, value=16)
        dropout = st.number_input('dropout rate', min_value=0.0, value=0.2)
        mlp_dims = st.text_input('MLP Network Dimensions (comma-separated)', '16,16')
        args += f" --embed_dim {embed_dim}"
        args += f" --mlp_dims {mlp_dims}"
        args += f" --dropout {dropout}"
    
    if model_choice in ('DeepFM'):
        # Add other NCF or cNCF specific options
        use_bn = st.radio('batch normalization', [True, False], index=1)
        merge_summary = st.radio('merge summary', [True, False], index=1)
        args += f" --use_bn {use_bn}"
        args += f" --merge_summary {merge_summary}"
        
        
    if model_choice in ('DCN'):
        num_layers = st.number_input('num_layer', value=3)
        args += f" --num_layers {num_layers}"
        
    
    if model_choice in ('CNN_FM'):
        cnn_embed_dim = st.number_input('CNN Embeddinbg Dimension', value=64)
        cnn_latent_dim = st.number_input('CNN latent Dimension', value=12)
        args += f" --cnn_embed_dim {cnn_embed_dim}"
        args += f" --cnn_latent_dim {cnn_latent_dim}"
        
    if model_choice in ('DeepCoNN'):
        vector_create = st.radio('Text Vector Creation', [True, False], index=1)
        deepconn_embed_dim = st.number_input("Embedding Dimension", min_value=1, value=32)
        deepconn_latent_dim = st.number_input("Latent Dimension", min_value=1, value=10)
        conv_1d_out_dim = st.number_input("1D Conv Output Dimension", min_value=1, value=50)
        kernel_size = st.number_input("Kernel Size", min_value=1, value=3)
        word_dim = st.number_input("Word Dimension", min_value=1, value=768)
        out_dim = st.number_input("Output Dimension", min_value=1, value=32)
        
        args += f" --vector_create {vector_create}"
        args += f" --deepconn_embed_dim {deepconn_embed_dim}"
        args += f" --deepconn_latent_dim {deepconn_latent_dim}"
        args += f" --conv_1d_out_dim {conv_1d_out_dim}"
        args += f" --kernel_size {kernel_size}"
        args += f" --word_dim {word_dim}"
        args += f" --out_dim {out_dim}"
        

    wandb_enabled = st.radio("Enable WandB", [True, False], index=0)
    args += f" --wandb {wandb_enabled}"
    
    if st.button("Train"):
        with st.spinner("Training models..."):
            subprocess.run(["python", "main.py"] + args.split())
        st.success("Training completed!")

def ensemble():
    st.title("Level-1 BookRatingPrediction")
    
    ensemble_files = st.text_input("Ensemble Files (comma-separated)", "")
    ensemble_strategy = st.selectbox("Ensemble Strategy", ['weighted', 'mixed'], index=0)
    ensemble_weight = st.text_input("Ensemble Weights (comma-separated)", "")
    args = ""
    args += f" --enseble_files {ensemble_files}"
    args += f" --ensemble_startegy {ensemble_strategy}"
    args += f" --ensebble_weight {ensemble_weight}"
    if st.button('Start Ensemble Learning'):
        with st.spinner("Waiting for Ensemble learning"):
            subprocess.run(["python", "ensemble.py"] + args.split())
        st.success("Ensemble Learning completed!")
     
if __name__ == '__main__':
    st.sidebar.title("Navigation")
    category = st.sidebar.selectbox("Go to", ["Train", "Ensemble"])
    
    if category == "Train":
        main()
    elif category == "Ensemble":
        ensemble()