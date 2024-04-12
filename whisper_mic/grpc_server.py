import grpc
import whisper_mic_service_pb2
import whisper_mic_service_pb2_grpc
import numpy as np
from concurrent import futures
from faster_whisper import WhisperModel


class WhisperMicServicer(whisper_mic_service_pb2_grpc.WhisperMicServicer):
    def StreamData(self, request_iterator, context):
        banned_results = {'', ' ', '\n', None}
        faster_whisper_model = WhisperModel('large-v3', device='cuda', compute_type='float16')

        for request in request_iterator:
            ndarray = np.frombuffer(request.ndarray_bytes, dtype=np.float32)
            print(f"Received data from client!")

            segments, _ = faster_whisper_model.transcribe(ndarray, language='ja')
            # del faster_whisper_model
            predicted_text = ''.join(segment.text for segment in segments)
            if predicted_text in banned_results:
                continue

            print(f'Predicted result: {predicted_text}')
            # Process the received data and generate a response
            yield whisper_mic_service_pb2.Transcription(prediction=predicted_text)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    whisper_mic_service_pb2_grpc.add_WhisperMicServicer_to_server(WhisperMicServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
