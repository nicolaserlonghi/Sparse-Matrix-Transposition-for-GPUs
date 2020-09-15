printf "\n#########################################################################################\n"
printf "#\t\t\t\t\t\t\t\t\t\t\t#\n"
printf "#\t\t\tRimozione del vecchio eseguibile\t\t\t\t#\n"
printf "#\t\t\t\t\t\t\t\t\t\t\t#\n"
printf "#########################################################################################\n"
rm -rf ./bin
printf "\n\n#########################################################################################\n"
printf "#\t\t\t\t\t\t\t\t\t\t\t#\n"
printf "#\t\tCompilazione del progetto e generazione dell'eseguibile\t\t\t#\n"
printf "#\t\t\t\t\t\t\t\t\t\t\t#\n"
printf "#########################################################################################\n"
make
printf "\n\n#########################################################################################\n"
printf "#\t\t\t\t\t\t\t\t\t\t\t#\n"
printf "#\t\tInstallazione dell'eseguibile nella cartella bin\t\t\t#\n"
printf "#\t\t\t\t\t\t\t\t\t\t\t#\n"
printf "#########################################################################################\n"
make install
printf "\n\n#########################################################################################\n"
printf "#\t\t\t\t\t\t\t\t\t\t\t#\n"
printf "#\t\t\t\tRimozione dei file .o creati\t\t\t\t#\n"
printf "#\t\t\t\t\t\t\t\t\t\t\t#\n"
printf "#########################################################################################\n"
make clean
printf "\n\n#########################################################################################\n"
printf "#\t\t\t\t\t\t\t\t\t\t\t#\n"
printf "#\t\t\t\tEsecuzione del programma\t\t\t\t#\n"
printf "#\t\t\t\t\t\t\t\t\t\t\t#\n"
printf "#########################################################################################\n"
cd testFiles
while read line;
do
    ../bin/sparse_matrix_transpose $line
    printf "\n\n"
done <<< "$(ls)"
printf "\n\n#########################################################################################\n"
printf "#\t\t\t\t\t\t\t\t\t\t\t#\n"
printf "#\t\t\t\t\tTerminato\t\t\t\t\t#\n"
printf "#\t\t\t\t\t\t\t\t\t\t\t#\n"
printf "#########################################################################################\n"