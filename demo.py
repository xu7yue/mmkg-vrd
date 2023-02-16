import os
from typing import List, Dict

from PIL import Image
import gradio as gr
from mmkg_vrd import VRD

pretrained_ckpt ='custom_io/ckpt/RX152FPN_reldn_oi_best.pth'
labelmap_file ='custom_io/ji_vrd_labelmap.json'
freq_prior ='custom_io/vrd_frequency_prior_include_background.npy'

def inference(input_image: str):
    
    vrd = VRD(pretrained_ckpt=pretrained_ckpt, labelmap_file=labelmap_file, freq_prior=freq_prior)
    
    dir_name = os.path.dirname(input_image)
    base_name = os.path.basename(input_image)
    ext = base_name.split('.')[-1]
    base_name_woext = '.'.join(base_name.split('.')[:-1])
    save_file = os.path.join(dir_name, f'{base_name_woext}.vrd.{ext}')
    
    vrd.det_and_vis(
        img_file=input_image,
        save_file=save_file
    )

    return save_file

def main():
    gr.Interface(
        inference,
        inputs=[
            gr.Image(type="filepath"),
        ],
        outputs=gr.Image(type="filepath"),
        examples=[
            [
                'custom_io/imgs/test0.jpg', 
            ],
            [
                'custom_io/imgs/test1.jpg', 
            ],
            [
                'custom_io/imgs/test2.png', 
            ],
            [
                'custom_io/imgs/test3.png', 
            ],
            [
                'custom_io/imgs/woman_fish.jpg', 
            ]
        ],
      
    ).launch(share=True)

if __name__ == "__main__":
    main()
