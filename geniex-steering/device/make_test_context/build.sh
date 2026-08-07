#!/bin/bash
# Builds the tiny steering-capable QNN context bundle for HTP v79 (SM8750)
# using the QAIRT 2.45.41 x86_64-linux tools. Run inside WSL as root.
set -e
QAIRT=/root/qairt-2.45
OUT=/root/steering-test-bundle
SRC="$(cd "$(dirname "$0")" && pwd)"

export PYTHONPATH=$QAIRT/lib/python
export PATH=$QAIRT/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$QAIRT/lib/x86_64-linux-clang

rm -rf $OUT && mkdir -p $OUT/work
python3 "$SRC/make_onnx.py" $OUT/work

PYBIN="${PYBIN:-/root/pyenv/bin/python}"
for g in prefill_ar4_cl16_1_of_1 token_ar1_cl16_1_of_1; do
  "$PYBIN" $QAIRT/bin/x86_64-linux-clang/qairt-converter --input_network $OUT/work/$g.onnx --output_path $OUT/work/$g.dlc
done

# HTP offline prepare for v79 / SM8750.
cat > $OUT/work/htp_config.json <<'EOF'
{
  "graphs": [ { "graph_names": ["prefill_ar4_cl16_1_of_1"], "vtcm_mb": 0, "O": 3 },
              { "graph_names": ["token_ar1_cl16_1_of_1"], "vtcm_mb": 0, "O": 3 } ],
  "devices": [ { "dsp_arch": "v79", "soc_model": 69 } ]
}
EOF
cat > $OUT/work/backend_ext.json <<EOF
{ "backend_extensions": {
    "shared_library_path": "$QAIRT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so",
    "config_file_path": "$OUT/work/htp_config.json" } }
EOF

qnn-context-binary-generator \
  --backend $QAIRT/lib/x86_64-linux-clang/libQnnHtp.so \
  --model libQnnModelDlc.so \
  --dlc_path $OUT/work/prefill_ar4_cl16_1_of_1.dlc \
  --binary_file prefill \
  --output_dir $OUT/work \
  --config_file $OUT/work/backend_ext.json

qnn-context-binary-generator \
  --backend $QAIRT/lib/x86_64-linux-clang/libQnnHtp.so \
  --model libQnnModelDlc.so \
  --dlc_path $OUT/work/token_ar1_cl16_1_of_1.dlc \
  --binary_file decode \
  --output_dir $OUT/work \
  --config_file $OUT/work/backend_ext.json

mkdir -p $OUT/bundle
cp $OUT/work/prefill.bin $OUT/work/decode.bin $OUT/bundle/
cp $OUT/work/aux_inputs.json $OUT/work/genie_config.json $OUT/work/embedding_weights.raw $OUT/bundle/
# Any parseable tokenizer satisfies the loader; forwardLogits never tokenizes.
cp "$1" $OUT/bundle/tokenizer.json
ls -la $OUT/bundle
echo BUNDLE_OK
