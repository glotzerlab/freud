DIRECTORIES=cpp/*
RETURNVAL=0
for f in $DIRECTORIES/*
do
    if ! diff <(clang-format -style=file $f) $f; then
        echo "$f has wrong formatting."
        RETURNVAL=1
    else
        echo "$f looks fine."
    fi
done
exit $RETURNVAL