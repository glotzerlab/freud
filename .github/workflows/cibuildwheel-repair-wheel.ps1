$DEST_DIR = $args[0]
$WHEEL = $args[1]

# Install delvewheel for Windows wheel repairs
pip install delvewheel==1.11.0

# From https://cibuildwheel.pypa.io/en/stable/options/#repair-wheel-command
delvewheel repair --add-path "C:/Program Files (x86)/TBB/bin/" -w "${DEST_DIR}" "${WHEEL}"
