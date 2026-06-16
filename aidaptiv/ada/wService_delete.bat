@echo off

set service_name=ada_service

sc.exe stop %service_name%
sc.exe delete %service_name%
