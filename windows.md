on my windows machine:
nvcc -o main .\main.cu -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.36.32532\bin\Hostx64\x64"; .\main.exe
<br>
or 
nvcc -o main --run .\main.cu -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.36.32532\bin\Hostx64\x64"


nvcc -o main --run src/glad.c .\main.cu -L./lib -I./include -lglfw3 -lopengl32 -lgdi32 -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.36.32532\bin\Hostx64\x64"
