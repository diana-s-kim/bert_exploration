#embedding layer access
import numpy as np
import torch
import torch.nn as nn
from Qformer import BertConfig, BertLMHeadModel
from transformers import BertTokenizer
from data.artemis import ArtEmis
from torch.utils.data import DataLoader
import pickle

device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")

def init_Qformer(num_query_token=32, vision_width=1408, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = False
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer


def init_tokenizer(truncation_side="right"):
       tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
       tokenizer.add_special_tokens({"bos_token": "[DEC]"})
       return tokenizer


class BERT_space(nn.Module):
        def __init__(self,embed_dim=256):
                super().__init__()
                self.tokenizer = init_tokenizer()   
                self.Qformer=init_Qformer()
                self.Qformer.resize_token_embeddings(len(self.tokenizer))
                self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
                self.color_texture_comp_head=nn.Linear(embed_dim,3)
                #here chop
#                self.embedding_1st=self.Qformer.bert
        def forward(self,X):
                return self.Qformer(X)

        def embedding(self,X):
                texts = X["text"]
                #append "texture", "composition", "color"
                text = [text+" texture composition color " for text  in texts]
                print("appended txt:",text)
                print(text)
                splited_txt=[self.tokenizer.tokenize(txt) for txt in text]
                print(splited_txt[0])
                text_tokens = self.tokenizer(text,truncation=True,max_length=512,return_tensors="pt").to(device)
                text_tokens_=text_tokens["input_ids"]
                attention_mask=text_tokens.attention_mask
                print("input_id",text_tokens_)
                print("input_txt",splited_txt)
                text_output=self.Qformer(text_tokens_)#sequential
                return splited_txt


def collect(dataloader, device, model):
        splited_txt=[]
        for batch, X in enumerate(dataloader):
                splited_txt.extend(model.embedding(X))
        return splited_txt
                
                
def main():
        bert=BERT_space().to(device)
        for name, param in bert.named_parameters():
                print ("before",name, param.data)

        #hook register        
        #activation={"attention":np.empty((0,12,32,32))}#lst append
        activation={"attention":[]}
        def getActivation(name):
                def hook(model, input, output):
#                    activation[name]=np.append(activation[name],output.cpu().data.numpy(),axis=0)
                     activation[name].append(output.cpu().data.numpy())# different length
                return hook
        bert.Qformer.bert.encoder.layer[0].attention.self.attn_prob.register_forward_hook(getActivation('attention'))

        #wihtout
        #dataloader
        artemis=ArtEmis(data_csv="/ibex/ai/home/kimds/Research/P2/withLLM/making_it_works/data/data_after.csv",img_dir="/ibex/ai/home/kimds/Research/P2/withLLM/making_it_works/img_after/")
        dataloader = DataLoader(artemis, batch_size=1, shuffle=False)
        
        #cllct
        results=collect(dataloader, device, bert)
        with open("./numpy_analysis_factor/splited_txt_wo.txt","w") as f:
                for item in results:
                        f.write(" ".join(item)+"\n")
        f.close()
        #np.save("./numpy_analysis_factor/output_attn_prob_wo.npy", attn=np.array(activation["attention"]),allow_pickle=True)
        with open("./numpy_analysis_factor/output_attn_prob_wo.pkl","wb") as f:
                pickle.dump(activation,f)

if __name__ == "__main__":
    main()
