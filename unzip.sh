echo "estrazione archivi\n"
for f in *.tar.gz; 
do
	pigz -dc "$f" | tar xf -
done
echo "eliminazione archivi\n"
for f in *.tar.gz;
do
	rm "$f"
done
echo "estrazione matrici\n"
file=$(ls -d */)
while read line; do mv "$line${line%?}.mtx" .; done <<< $file
echo "eliminazione cartelle\n"
while read line; do rm -rf $line; done <<< $file

