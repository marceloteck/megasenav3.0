@echo off
rem Ativar o ambiente virtual 
call Scripts\activate

rem Executar o script Python
python MEGA-SENA\MegaSena_static.1.2.py

rem Manter a janela do terminal aberta
pause
