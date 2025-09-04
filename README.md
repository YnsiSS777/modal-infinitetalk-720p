# InfiniteTalk on Modal

This project deploys the [MeiGen-AI/InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) model on [Modal](https://modal.com) to provide a high-performance, scalable API for generating talking head videos from an image and audio files.

The deployment is optimized for efficient inference, leveraging:

- L40s GPUs (can be easily switched to other [Modal GPU types](https://modal.com/docs/guide/gpu#specifying-gpu-type))
- `FusioniX` LoRA optimization
- `flash-attention`
- `teacache`

Note that this does not fully implement the multi-speaking capabilities since that requires downloading separate model weights. We focus on single person for now. Open an issue if you would like multi-person support.

## Prerequisites

1. **Clone this Repository:**

   ```bash
   git clone https://github.com/Square-Zero-Labs/modal-infinitetalk
   cd modal-infinitetalk
   ```

2. **Create a Modal Account:** Sign up for a free account at [modal.com](https://modal.com).

3. **Install Modal Client:** Install the Modal client library and set up your authentication token.
   ```bash
   pip install modal
   modal setup
   ```

## Deployment

The application consists of a persistent web endpoint for production use and a local CLI for testing. It uses a `Volume` to cache the large model files, ensuring they are only downloaded once. A second `Volume` is used to efficiently handle the video outputs.

To deploy the web endpoint, run the following command from your terminal:

```bash
pip install pydantic
modal deploy app.py
```

Modal will build a custom container image, download the model weights into a persistent `Volume`, and deploy the application. After a successful deployment, it will provide a public URL for your API endpoint.

The initial deployment will take several minutes as it needs to download the large model files. Subsequent deployments will be much faster as the models are cached in the volume.

The weights are under the volume name `infinitetalk-models`.
Output videos are saved under the volume name `infinitetalk-outputs`.

## Usage

### 1. Local Testing CLI

For development and testing, you can use the built-in command-line interface to run inference on local files or URLs.

```bash
modal run app.py --image-path "https://example.com/portrait.jpg" --audio1-path "https://example.com/audio.mp3" --prompt "A dog talking" --output-path outputs/my_video.mp4
```

### 2. Web API Endpoint

The deployed service can be called via a `POST` request with proxy authentication. The API accepts a JSON payload with the following fields:

- `image` (string, required): A URL to the source image **or video**. The input can be image or video. If video the output video will contain movement from the input video adjusted with the new lip sync to the audio.
- `audio1` (string, required): A URL to the source audio file (MP3 or WAV).
- `prompt` (string, optional): A text prompt.

The duration of the output video is automatically determined by the length of the input audio. The video frame count is calculated to match this combined duration while adhering to the model's `4n+1` frame constraint.

#### Authentication

The API requires proxy authentication tokens.

To create proxy auth tokens, go to your Modal workspace settings and generate a new token. Set the token ID and secret as environment variables:

```bash
export TOKEN_ID="your-token-id"
export TOKEN_SECRET="your-token-secret"
```

**API Usage - Polling Pattern:**

Following [Modal's recommended polling pattern](https://modal.com/docs/guide/webhook-timeouts), we use two endpoints for long-running video generation:

1. **Submit Job** - Starts generation and returns call_id
2. **Poll Results** - Check status and download video when ready

**Step 1: Submit Job**

```bash
# Submit video generation job and capture call_id
CALL_ID=$(curl -s -X POST \
     -H "Content-Type: application/json" \
     -H "Modal-Key: $TOKEN_ID" \
     -H "Modal-Secret: $TOKEN_SECRET" \
     -d '{
           "image": "https://example.com/portrait.jpg",
           "audio1": "https://example.com/audio.mp3",
           "prompt": "A dog is talking"
         }' \
     "https://<username>--infinitetalk-api-model-submit.modal.run" | jq -r '.call_id')

echo "Job submitted with call_id: $CALL_ID"
```

**Step 2: Poll for Results**

```bash
# Poll for results - downloads video on success, shows status on failure
HTTP_STATUS=$(curl -w "%{http_code}" -s --head \
     -H "Modal-Key: $TOKEN_ID" \
     -H "Modal-Secret: $TOKEN_SECRET" \
     "https://<username>--infinitetalk-api-api-result-head.modal.run?call_id=$CALL_ID")
echo "HTTP $HTTP_STATUS"
```

- `202 Accepted` - Shows processing status in terminal only
- `200 OK` - Downloads video to `outputs/generated_video.mp4`

**Step 3: Retrieve Finished Video**

```bash
curl -X GET \
         -H "Modal-Key: $TOKEN_ID" \
         -H "Modal-Secret: $TOKEN_SECRET" \
         --output outputs/api-generated_video.mp4 \
         "https://<username>--infinitetalk-api-api-result.modal.run?call_id=$CALL_ID"
    echo "Video saved to outputs/api-generated_video.mp4"
```

Replace:

- `<username>` with your actual Modal username
- `$TOKEN_ID` and `$TOKEN_SECRET` with your proxy auth token credentials

The URL format is `https://[username]--[app-name]-[class-name]-[method-name].modal.run` where the class-name is `model` or `api` and the method-name are the methods defined in app.py.

## Resources

- [InfiniteTalk Paper](https://arxiv.org/pdf/2508.14033)
- [InfiniteTalk Repo](https://github.com/MeiGen-AI/InfiniteTalk)

## Development Notes

### Git Subtree Management

When originally added:

```bash
git subtree add --prefix infinitetalk https://github.com/MeiGen-AI/InfiniteTalk main --squash
```

If the original `InfiniteTalk` repository is updated and you want to incorporate those changes into this project, you can pull the updates using the following command:

```bash
git subtree pull --prefix infinitetalk https://github.com/MeiGen-AI/InfiniteTalk main --squash
```

### Changes made to InfiniteTalk base repo

- generate_infinitetalk.py (attention fix)
