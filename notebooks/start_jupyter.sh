ARG1=${1:-9999}
jupyter notebook \
--ip=0.0.0.0 \
--port=$ARG1 \
--allow-root \
--NotebookApp.token=''