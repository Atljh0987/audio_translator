import os
import sys
from typing import List
import gigaam
import subprocess

# Hugged Face token must be set
# os.environ["HF_TOKEN"] = ""

ext_processor = {
    ".wav": lambda file_obj : translateWav(file_obj),
    ".mp4": lambda file_obj : translateMp4(file_obj)
}

model = gigaam.load_model("v2_rnnt")

class FileInfo:
    def __init__(self, full_path):
        self.full_path = full_path
        self.dir_path = os.path.dirname(full_path) 
        self.file_name = os.path.basename(full_path)  
        self.file_ext = extractExtension(self.file_name).lower()
        self.file_base = os.path.splitext(self.file_name)[0]

def main():
    if len(sys.argv) < 2:
        print("You must use this: python script.py <folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]

    file_objects: List[FileInfo] = recursive_file_scan(folder_path)    

    for file_obj in file_objects:
        result = ext_processor[file_obj.file_ext](file_obj)

        if result is None:
            continue
        
        saveToFile(result, f'{file_obj.dir_path}/{file_obj.file_name}_translation.txt')

    print(f"\nThe End")

def recursive_file_scan(directory) -> List[FileInfo]:
    file_list: List[FileInfo] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if correctFile(file):
                full_path = os.path.join(root, file)                
                file_list.append(FileInfo(full_path))
                
    return file_list

def correctFile(file):
    return extractExtension(file) in ext_processor

def extractExtension(file):
    return os.path.splitext(file)[1]

def translateWav(file_obj: FileInfo):
    try:
        return model.transcribe_longform(file_obj.full_path)
    except Exception as e:
        print(e)
        return None
    
def translateMp4(file_obj: FileInfo):
    temp_wav = os.path.join(file_obj.dir_path, f'temp_{file_obj.file_base}.wav')

    if convert_mp4_to_wav(file_obj.full_path, temp_wav):
        try:
            return translateWav(FileInfo(temp_wav))
        finally:
            os.remove(temp_wav)
    else:
        return None
   
            
def convert_mp4_to_wav(mp4_path, wav_path):
    try:
        subprocess.run([
            'ffmpeg',
            '-i', mp4_path,
            '-acodec', 'pcm_s16le',
            '-ac', '1',
            '-ar', '11025',
            wav_path
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Conversation error {mp4_path}: {e}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg didn't found. Please install ffmpeg for video conversation.")
        return False

def saveToFile(result, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for utterance in result:
            transcription = utterance["transcription"]
            start, end = utterance["boundaries"]
        
            line = f"[{gigaam.format_time(start)} - {gigaam.format_time(end)}]: {transcription}\n"
        
            f.write(line)
            print(line, end="")

    print(f"\nResult saved in file: {output_file}")

if __name__ == "__main__":
    main()