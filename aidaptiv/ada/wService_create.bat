@echo off

set service_name=ada_service

if "%~1"=="" (goto USE_BATCH_PATH)

:USE_PARMS
if exist "%~1%\ada.exe" (
	set ada_path=%~1%\ada.exe
	goto DONE
)

:USE_BATCH_PATH
if exist "%~dp0\ada.exe" (
	set ada_path=%~dp0\ada.exe
	goto DONE
)

:USE_CD
if exist "%cd%\ada.exe" (
	set ada_path=%cd%\ada.exe
	goto DONE
)

echo ada.exe not found
pause
goto EOF

:DONE
echo ada = %ada_path%

sc.exe create %service_name% binPath="%ada_path%" start=auto && sc.exe start %service_name%
