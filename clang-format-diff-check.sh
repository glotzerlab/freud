DIRECTORIES=cpp/*
RETURNVAL=0
for f in $DIRECTORIES/*
do
    if ! diff <(clang-format -style=file $f) $f; then
        echo "$f has wrong formatting"
        RETURNVAL=1
    fi
done
exit $RETURNVAL