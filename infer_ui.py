import gradio as gr
import os
import subprocess
import glob
import os
import os
import yaml



spk_list = None
will_use_moudel = None

all_model_list = None
if not os.path.exists("./results"):
    os.makedirs("./results")
    


def main_ui():
     #在这里声明全局变量
     global spk_list
     global all_model_list
     global will_use_moudel
     with gr.Blocks() as ui:
        gr.Markdown('# Diffusion-SVC-infer-WebUI-bate1')
        gr.Markdown('这里是一个markdown框框，里面的内容也来自变量，使用时切记声明为全局变量')
        with gr.Tab("先读一读喵~"):
            with gr.Accordion('这是一个抽屉',open=False):
                gr.Markdown('这里的内容将会从变量获取')
            gr.Markdown('这里不是一个抽屉，这里的内容也来自变量')
        
        with gr.Tab("推理"):
            with gr.Row():
                get_all_moudel=gr.Button(value='刷新')
            with gr.Row():
                all_moudel=gr.Dropdown(choices=all_model_list,type='value',label='推理模型位置',interactive=True)    
            with gr.Row():
                get_new_list=gr.Button(value='确定')
            with gr.Row():
                spk=gr.Dropdown(choices=spk_list,label='说话人',type='index',interactive=True,allow_custom_value=True)
                will_use_moudel=gr.Dropdown(choices=will_use_moudel,type='value',label='推理模型',interactive=True)    

            with gr.Row():
                input_wav=gr.Audio(type='filepath',label='选择待转换音频')
            
            with gr.Row():
                keychange=gr.Slider(-24,24,value=0,step=1,label='变调')
                infer_step=gr.Number(value=20,label='inferstep',info='采样步数')
            with gr.Row():
                method=gr.Dropdown(['rk4','euler'],value='euler',type='value',label='采样器',interactive=True)
                f0=gr.Dropdown(['parselmouth', 'dio', 'harvest', 'crepe','rmvpe','fcpe'],type='value',value='rmvpe',label='f0提取器种类',interactive=True)
                t_start=gr.Slider(0,1,value=0.7,step=0.1,label='ts',interactive=True)
            with gr.Row():
                bt_infer=gr.Button(value='开始转换')
                output_wav=gr.Audio(type='filepath',label='输出音频')
            get_new_list.click(fn=GetSpkList,inputs=[all_moudel],outputs=[spk,will_use_moudel])
            get_all_moudel.click(fn=get_all_moudel1,inputs=[],outputs=all_moudel)
            bt_infer.click(fn=inference,inputs=[input_wav,keychange,spk,infer_step,method,f0,all_moudel,will_use_moudel,t_start],outputs=output_wav)
          

        ui.launch(inbrowser=True)










def get_all_moudel1():


    def list_direct_subdirectories(paths):

        all_subdirectories = []
        for path in paths:
            # 使用listdir而非walk，以仅获取当前目录下的项
            for name in os.listdir(path):
                full_path = os.path.join(path, name)
                # 确保是目录且不是指向父目录或自身的符号链接
                if os.path.isdir(full_path) and not os.path.samefile(full_path, path):
                    # 使用relpath获取相对于初始路径的子目录路径
                    relative_path = os.path.relpath(full_path, path)
                    all_subdirectories.append(relative_path)
        return all_subdirectories
    all_path=['./workdir']
    all_model_list = list_direct_subdirectories(all_path)
    return gr.update(choices=all_model_list)



def inference(input_wav:str,keychange,spk:int,infer_step,method,f0,all_moudel:str,use_model:str,t_start:float):
        
        model_path="./workdir/"+all_moudel+"/exp/diffusion/"+use_model
        
        id=spk+1
        print(input_wav,model_path,id)
        output_wav='results/'+ input_wav.replace('\\','/').split('/')[-1]
        input_wav='"'+input_wav+'"'
        cmd='python -u F:/ReFlow-VAE-SVC-main/main.py -i '+input_wav+' -m '+str(model_path)+' -o '+output_wav+' -k '+str(int(keychange))+' -tid '+str(int(id))+' -method '+method+' -pe '+f0+' -step '+str(int(infer_step))+' -tstart '+str(t_start)
        #cmd='python -u main.py -i '+input_wav+' -m '+model_path+' -o '+output_wav+' -k '+str(int(keychange))+' -tid '+str(int(id))+' -method '+method+' -step '+str(int(infer_step))
        print(cmd)
        os.system(cmd)
        print('推理完成')
        return output_wav

def GetSpkList(use_path:str):
        
        directory="./workdir/"+use_path+"/exp/diffusion"
        path=directory+"/config.yaml"
        path=str(path)
        print(path)
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        spk_list = data.get('spks', [])
        print(spk_list) 


        
        extension=".pt"
    # 使用glob模块匹配指定目录下指定后缀的文件
        pattern = os.path.join(directory, f'*{extension}')
        files = glob.glob(pattern)
    
    # 从完整路径中提取文件名
        file_names = [os.path.basename(file) for file in files]
        

        
        return gr.update(choices=spk_list),gr.update(choices=file_names)



main_ui()