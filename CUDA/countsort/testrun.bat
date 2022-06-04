SET output=.\outputs.txt

IF EXIST %output% DEL /F %output%

make

::x -> input size
FOR %%x IN ( 100000 200000 300000 400000 ) DO (
    ::y -> number of blocks
    FOR %%y IN ( 10 50 100 ) DO (
        ::z -> threads per block
        FOR %%z IN ( 10 50 100 ) DO (
            .\countsort.exe %%x %%y %%z >> %output%
            timeout /t 1
        )
    )
)

make clean

