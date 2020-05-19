HADOOP_EXE='/usr/bin/hadoop'
HADOOP_LIBPATH='/opt/cloudera/parcels/CDH/lib'
HADOOP_STREAMING='hadoop-mapreduce/hadoop-streaming.jar'

alias hfs="$HADOOP_EXE fs"
alias hjs="$HADOOP_EXE jar $HADOOP_LIBPATH/$HADOOP_STREAMING"

module load python/gnu/3.6.5
module load spark/2.4.0

alias spark-submit='PYSPARK_PYTHON=$(which python) spark-submit'

