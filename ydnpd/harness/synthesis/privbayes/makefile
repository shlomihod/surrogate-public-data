UNAME := $(shell uname)

all:
	$(info UNAME is $(UNAME))
	if [ "$(UNAME)" = "Linux" ]; then \
		g++ -O3 --std=c++14 code/main_marginal.cpp code/methods.cpp code/noise.cpp code/printer.cpp code/table.cpp code/translator.cpp -o privBayes.bin -lboost_random; \
	else \
		g++ -O3 --std=c++14 code/main_marginal.cpp code/methods.cpp code/noise.cpp code/printer.cpp code/table.cpp code/translator.cpp -o privBayes.bin -lboost_random-mt; \
	fi
