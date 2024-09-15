import gradio as gr 
import os
import glob
import yaml
import shutil
import subprocess
import tqdm
import librosa
import soundfile as sf
import scipy
from slicer2 import Slicer
import fnmatch
from pydub import AudioSegment
import concurrent.futures


#检查目录存在性
if not os.path.exists("./results"):
    os.makedirs("./results")

if not os.path.exists("./backup"):
    os.makedirs("./backup")
if not os.path.exists("./exp"):
    os.makedirs("./exp")
if not os.path.exists("./workdir"):
    os.makedirs("./workdir")



def main_ui():
    config_list=None
    with gr.Blocks() as ui:
        gr.Markdown('# Diffusion-SVC-Webui-1.0.0bate1')
        
        with gr.Tab("创建工程"):
            gr.Markdown('创建工程进度请关注控制台！！！工程创建成功后因故重启后端后只需要填入工程名 <br> 你需要把你的训练集手动放到workdir下的dataset_tmp里面')
            with gr.Row():
                work_name=gr.Textbox(label="工程名",lines=1, placeholder="输入工程名",interactive=True)
            with gr.Row():    
                config_in_create_config=gr.Dropdown(choices=config_list,type='value',label='工程配置文件使用',interactive=True)
                get_all_config_list=gr.Button(value='刷新配置文件')
            
            with gr.Row():
                create_wordir_bt=gr.Button(value='创建工程')
            get_all_config_list.click(fn=get_config_list,outputs=config_in_create_config)
            create_wordir_bt.click(fn=create_wordir,inputs=[config_in_create_config,work_name])
        with gr.Tab("训练/微调"):
            with gr.Tab("数据预处理"):
                with gr.Row():
                    audio_process_bt=gr.Button(value='开始数据前置处理')
                with gr.Row():
                    process_f0=gr.Dropdown(['parselmouth', 'dio', 'harvest', 'crepe','rmvpe','fcpe'],type='value',value='rmvpe',label='f0提取器种类',interactive=True)
                with gr.Row():    
                    bt_auto_process_start=gr.Button(value='开始正式预处理')
                audio_process_bt.click(fn=audio_process,inputs=[work_name])
                bt_auto_process_start.click(fn=audio_process_all,inputs=[work_name,process_f0])
            with gr.Tab("训练"):
                gr.Markdown('这里的配置项目修改不全，请自行修改你工程目录下的配置文件<br>如果需要底模，请自行放入工程下的exp/diffusionsvc内')
                
                with gr.Row():
                    batch_size=gr.Slider(minimum=12,maximum=1024,value=24,label='Batch_size',interactive=True)
                    learning_rate=gr.Number(value=0.0005,label='学习率',info='真心不建议超过0.0002',interactive=True)
                    val=gr.Number(value=1000,label='验证步数',info='检查点，会被覆盖哦',interactive=True)
                    save=gr.Number(value=1000,label='强制保存步数',info='被强制保存的模型',interactive=True)
                    
                with gr.Row():
                    amp_dtype=gr.Dropdown(['fp16','fp32',"bf16"],value='fp16',label='训练精度',interactive=True)
                    interval_log=gr.Number(value=1,label='interval_log',info='评估日志步数',interactive=True)
                    decay_step=gr.Number(value=1000,label='decay_step',info='学习率衰减的步数',interactive=True)
                    gamma=gr.Number(value=0.5,label='gamma',info='学习率衰减的量',interactive=True)                    
                        
                with gr.Row():
                    num_workers=gr.Number(value=2,label='读取数据进程数',info='如果你的设备性能很好，可以设置为0',interactive=True)
                    cache_all_data=gr.Checkbox(value=True,label='启用缓存',info='将数据全部加载以加速训练',interactive=True)
                    cache_device=gr.Dropdown(['cuda','cpu'],value='cuda',type='value',label='缓存设备',info='如果你的显存比较大，设置为cuda',interactive=True)
                with gr.Row():    
                    bt_train_config=gr.Button(value='写入配置文件')
                    bt_train=gr.Button(value='开始训练')

                bt_train_config.click(fn=change_train_config,inputs=[num_workers,amp_dtype,batch_size,learning_rate,decay_step,cache_device,cache_all_data,interval_log,val,save,gamma,work_name])
                bt_train.click(fn=train,inputs=[work_name])
    ui.launch(inbrowser=True)




def train(dir_name):
    config_path="./workdir/"+dir_name+'/config.yaml'
    subprocess.run('start cmd /k python -u train.py -c '+config_path,shell=True,stdout=subprocess.PIPE)









def change_train_config(num_workers,amp_dtype,batch_size,lr,decay_step,cache_device,cache_all_data,interval_log,interval_val,interval_force_save,gamma,dir_name):
    
        
    with open("./workdir/"+dir_name+'/config.yaml', "r") as ref:
        existing_config = yaml.safe_load(ref)
    
    existing_config["train"]["num_workers"] = int(num_workers)
    existing_config["train"]["amp_dtype"] = str(amp_dtype)
    existing_config["train"]["batch_size"] = int(batch_size)
    existing_config["train"]["lr"] = float(lr)
    existing_config["train"]["decay_step"] = int(decay_step)
    existing_config["train"]["cache_device"] = str(cache_device)
    existing_config["train"]["cache_all_data"] = str(cache_all_data)
    existing_config["train"]["interval_log"] = int(interval_log)
    existing_config["train"]["interval_val"] = int(interval_val)
    existing_config["train"]["interval_force_save"] = int(interval_force_save)
    existing_config['train']['gamma']=gamma

    with open("./workdir/"+dir_name+'/config.yaml', "w") as ref:
        yaml.dump(existing_config, ref)



def audio_process_all(dir_name,process_f0):
    path='./workdir/'+dir_name+'/data/train/audio'
    spk_numb=len([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])
    with open("./workdir/"+dir_name+'/config.yaml', 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        data['data']['f0_extractor']=str(process_f0)
        data["model"]["n_spk"] = spk_numb
        with open("./workdir/"+dir_name+'/config.yaml','w',encoding='utf-8') as f:
            yaml.dump(data,f)
    config_path="./workdir/"+dir_name+'/config.yaml'
     
    subprocess.run('start cmd /k python -u preprocess.py -c '+config_path,shell=True,stdout=subprocess.PIPE)    
    





def audio_process(dir_name):
    print("开始重采样")
    audio_resp("./workdir/"+dir_name+'/dataset_tmp','./workdir/'+dir_name+'/data/train/audio')
    print('开始切片过长音频')

    

    slice_and_remove_original('./workdir/'+dir_name+'/data/train/audio', 20000)
    print('开始删除过短音频')

    delete_short_wav_files('./workdir/'+dir_name+'/data/train/audio', 2000)
    print('开始进行spk对照')
    append_original_names_to_config('./workdir/'+dir_name+'/data/train/audio',"./workdir/"+dir_name+'/config.yaml',['sliced'])
    print('开始进行验证集选择')
    if not os.path.exists('./workdir/'+dir_name+'/data/val/audio'):
        os.makedirs('./workdir/'+dir_name+'/data/val/audio')
    move_files_with_structure('./workdir/'+dir_name+'/data/train/audio', './workdir/'+dir_name+'/data/val/audio', 2)







def get_config_list():
    
    directory='./configs'
    yaml_files_full_path = glob.glob(os.path.join(directory, '**', '*.yaml'), recursive=True)

# 创建一个只包含文件名的新列表
    config_list = [os.path.basename(file) for file in yaml_files_full_path]
    return gr.update(choices=config_list)




def create_wordir(config_name,dir_name):
    if  os.path.exists("./workdir/"+dir_name):
        print('存在同名文件，为了保证数据安全工程创建已经终止,如果是由于创建失败而重新创建，请手动前往workdir删除创建失败的工程')
    else:
        os.makedirs("./workdir/"+dir_name)
        with open("./configs/"+config_name, "r") as ref:
            data = yaml.safe_load(ref)
            data['data']['train_path']=str("./workdir/"+dir_name+'/data/train')
            data['data']['valid_path']=str("./workdir/"+dir_name+'/data/val')
            data['env']['expdir']=str("/workdir/"+dir_name+'/exp/diffusionsvc')
        with open("./workdir/"+dir_name+'/config.yaml', 'w', encoding='utf-8') as file:
            yaml.dump(data, file, default_flow_style=False)
        os.makedirs("./workdir/"+dir_name+'/data')
        os.makedirs("./workdir/"+dir_name+'/dataset_tmp')
        #print('开始创建训练所需的文件，这一步可能耗时较长（取决于硬盘）')
        #shutil.copy('./dataset_raw', "./workdir/"+dir_name+'/dataset_tmp')
        print('工程创建完成')





def audio_resp(input_dir,output_dir):#这个代码块进行转码和重采样
    def process_file(file_path):
        # 使用pydub加载音频文件
        audio = AudioSegment.from_file(file_path)
        
        # 获取文件名和相对路径
        base_name = os.path.basename(file_path)
        name_without_ext, ext = os.path.splitext(base_name)
        relative_path = os.path.relpath(os.path.dirname(file_path), input_dir)
        
        # 输出文件路径
        output_subdir = os.path.join(output_dir, relative_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        output_file_path = os.path.join(output_subdir, f"{name_without_ext}.wav")
        
        # 直接覆盖已存在文件
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        
        # 重采样至44100Hz并转换为wav格式
        audio = audio.set_frame_rate(44100)
        audio.export(output_file_path, format="wav")
        
        print(f"转码和重采样 {file_path} -> {output_file_path}")

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 使用遍历并收集所有音频文件路径
    audio_files = []
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(('.ogg', '.wav', '.flac','.mp3')):
                audio_files.append(os.path.join(root, filename))

    # 使用多线程
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file) for file in audio_files]
        for future in concurrent.futures.as_completed(futures):
            pass  









def slice_and_remove_original(directory_path, min_duration_ms):
    
    slicer_params = {
        'sr': 44110,          # 采样率自动检测
        'threshold': -40,   # 静音判定阈值
        'min_length': 5000, # 切片最小长度（毫秒）
        'min_interval': 300,# 静默间隔阈值（毫秒）
        'hop_size': 10,     # 帧跳跃大小（毫秒）
        'max_sil_kept': 500# 切片周围保留的最大静音长度（毫秒）
    }
    slicer = Slicer(**slicer_params)

    # 遍历目录
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                
                # 检查文件时长是否超过最短处理时长
                audio, sr = librosa.load(file_path, sr=None, mono=False)
                duration_ms = len(audio) / sr * 1000
                if duration_ms > min_duration_ms:
                    print(f"正在处理文件: {file_path}")
                    
                    # 执行音频切片
                    chunks = slicer.slice(audio)
                    
                    # 为每个切片创建新的文件名，并保存到原目录
                    base_name, ext = os.path.splitext(file)
                    for i, chunk in enumerate(chunks):
                        if len(chunk.shape) > 1:
                            chunk = chunk.T  # 对于立体声音频交换轴
                        output_file_path = os.path.join(root, f"{base_name}_{i}{ext}")
                        sf.write(output_file_path, chunk, sr)
                        print(f"已保存切片音频至: {output_file_path}")
                    
                    # 删除源文件
                    try:
                        os.remove(file_path)
                        print(f"已删除源文件: {file_path}")
                    except:
                        print(f"已跳过文件: {file_path}")







def append_original_names_to_config(dir_path, config_path, exclude_folders=None):
    if exclude_folders is None:
        exclude_folders = []

    # 初始化原名列表
    original_names = []
    counter = 1
    
    # 遍历指定目录
    for root, dirs, _ in os.walk(dir_path):
        # 过滤掉需要排除的文件夹
        dirs = [d for d in dirs if d not in exclude_folders]
        
        for dir_name in dirs:
            old_path = os.path.join(root, dir_name)
            new_name = str(counter)  # 使用数字作为新名称
            new_path = os.path.join(root, new_name)
            
            # 重命名文件夹
            os.rename(old_path, new_path)
            print(f"将 '{dir_name}' 作为说话人 '{new_name}'")
            
            # 仅记录原名
            original_names.append(dir_name)
            counter += 1
    
    # 读取现有配置
    try:
        with open(config_path, 'r') as config_file:
            config_data = yaml.safe_load(config_file)
    except FileNotFoundError:
        config_data = {}

    # 更新或添加数据
    if "spks" not in config_data:
        config_data["spks"] = []
    config_data["spks"].extend(original_names)

    # 写回配置文件
    with open(config_path, 'w') as config_file:
        yaml.dump(config_data, config_file, default_flow_style=False)
    
    print("写入配置文件完成")



def delete_short_wav_files(directory, min_length_ms):
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.wav'):
            file_path = os.path.join(root, filename)
            try:
                audio = AudioSegment.from_file(file_path, format="wav")
                if len(audio) < min_length_ms:
                    print(f"删除音频，因为其太短: {file_path}")
                    try:
                        os.remove(file_path)
                    except PermissionError:
                        # 如果os.remove()因权限问题失败，尝试使用系统命令
                        if os.name == 'nt':  # Windows
                            subprocess.run(['del', '/F', file_path], shell=True, check=True)
                        else:  # Unix/Linux
                            subprocess.run(['rm', '-f', file_path], check=True)
            except Exception as e:
                print(f"删不了，怎么想都删不了吧: {file_path}. Error: {str(e)}")


def create_matching_structure(src_dir, dst_dir):

    for root, dirs, _ in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        dst_root = os.path.join(dst_dir, rel_path)
        if not os.path.exists(dst_root):
            os.makedirs(dst_root)

def move_files_with_structure(src_dir, dst_dir, num_files=1):
    
    create_matching_structure(src_dir, dst_dir)

    
    for subdir, _, files in os.walk(src_dir):
        
        moved_count = 0
        for file in files:
            if moved_count >= num_files:
                break
            
            
            src_path = os.path.join(subdir, file)
            
            
            rel_path = os.path.relpath(subdir, src_dir)
            dst_subdir = os.path.join(dst_dir, rel_path)
            dst_path = os.path.join(dst_subdir, file)
            
            
            shutil.move(src_path, dst_path)
            print(f"Moved {file} to {dst_path}")
            moved_count += 1







main_ui()