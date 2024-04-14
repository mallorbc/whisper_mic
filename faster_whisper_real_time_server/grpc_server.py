from concurrent import futures

import click
import grpc
import numpy as np
import torch
from faster_whisper import WhisperModel

import faster_whisper_transcription_pb2
import faster_whisper_transcription_pb2_grpc

faster_whisper_model = object()


class WhisperMicServicer(faster_whisper_transcription_pb2_grpc.FasterWhisperTranscription):
    def StreamData(self, request_iterator, context):  # pylint: disable=arguments-differ, unused-argument
        banned_results = {'', ' ', '\n', None}

        for request in request_iterator:
            ndarray = np.frombuffer(request.ndarray_bytes, dtype=np.float32)
            print('Received data from client!')

            segments, _ = faster_whisper_model.transcribe(ndarray, language=None if request.language == '' else request.language)
            predicted_text = ''.join(segment.text for segment in segments)
            if predicted_text in banned_results:
                continue

            print(f'Predicted result with language {request.language}: {predicted_text}')
            yield faster_whisper_transcription_pb2.Transcription(prediction=predicted_text)


# pylint: disable=no-value-for-parameter
@click.command()
@click.option('--model', default='large-v3', help='Model (`distil-large-v3` if en only)', type=click.Choice(['medium', 'large-v3', 'distil-large-v3']))
@click.option('--device', default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device', type=click.Choice(['cpu', 'cuda']))
@click.option('--precision', default='float16', help='Precision level', type=click.Choice(['int8', 'float16']))
def serve(
    model: str,
    device: str,
    precision: str,
):
    print(f'[INFO] Run in {device} on {model} model with {precision} precision.')
    global faster_whisper_model
    faster_whisper_model = WhisperModel(model_size_or_path=model, device=device, compute_type=precision)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    faster_whisper_transcription_pb2_grpc.add_FasterWhisperTranscriptionServicer_to_server(WhisperMicServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
