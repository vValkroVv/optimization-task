from llmfoundry.data.collators.audio_collator import AudioCollator
from llmfoundry.data.processors.base_processor import DataModality


class VideoAudioCollator(AudioCollator):
    """Soundtracks of `video_audio` files: the `AudioCollator` payload, in the
    `video_audio` bundle next to the frames' `video` bundle."""

    data_modality = DataModality.VIDEO_AUDIO
