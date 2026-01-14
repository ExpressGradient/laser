export LITELLM_MODEL=glm-4.7
export LITELLM_PROXY_API_KEY=sk-1234
export LITELLM_PROXY_API_BASE=http://192.168.43.179:4001
uv run laser --model litellm/glm-4.7 $@

exit

o use:

$: uv tool install git+https://github.com/ExpressGradient/laser
$: laser

or 
$: uvx --from git+https://github.com/ExpressGradient/laser laser