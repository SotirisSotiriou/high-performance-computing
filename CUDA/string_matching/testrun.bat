SET output=.\outputs.txt
SET filename=.\input.txt
SET filesize=1000000000
SET pattern=pi

IF EXIST %output% DEL /F %output%
IF EXIST %filename% DEL /F %filename%

make

.\produce.exe %filename% %filesize%

::x -> number of blocks
FOR %%x IN ( 10 20 30 ) DO (
    ::y -> threads per block
    FOR %%y IN ( 10 50 100 ) DO (
        .\string_matching.exe %filename% %pattern% %%x %%y >> %output%
        timeout /t 1
    )
)

IF EXIST %filename% DEL /F %filename%

make clean