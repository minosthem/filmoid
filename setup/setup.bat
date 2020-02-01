@ECHO OFF

:: go to parent directory
cd ..

:: create necessary folders
mkdir "%OUTPUT_FOLDER%"
mkdir "%RESOURCES_FOLDER%"
mkdir "%DATASETS_FOLDER%"

:: download glove file
cd "%RESOURCES_FOLDER%"
./wget.exe "%EMBEDDINGS_FILE_URL%"
./7zip.exe e "%EMBEDDINGS_FILE%"
cd ..

:: download datasets
cd "%DATASETS_FOLDER%"
./wget.exe http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
./wget.exe http://files.grouplens.org/datasets/movielens/ml-latest.zip

./7zip.exe e ml-latest-small.zip
./7zip.exe e ml-latest.zip

del ml-latest-small.zip
del ml-latest.zip

PAUSE