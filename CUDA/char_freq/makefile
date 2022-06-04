ifeq ($(OS),Windows_NT)
	RM=del
	export_files=char_freq.exe char_freq.lib char_freq.exp produce.exe
endif

ifeq ($(OS),Linux)
	RM=rm -f
	export_files=char_freq char_freq.lib char_freq.exp produce
endif

char_freq: char_freq.cu produce_data.c
	gcc -o produce produce_data.c
	nvcc -o char_freq char_freq.cu

clean:
	$(RM) $(export_files)