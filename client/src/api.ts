import { OutputParams } from './params';

/**
 * Uploads a video file to the Flask API and returns decoded output.
 * @param videoFile The File object from a file input element.
 */
export const decode = async (videoFile: File): Promise<OutputParams> => {
  const formData = new FormData();
  formData.append('video', videoFile);

  const response = await fetch('http://localhost:5000/decode', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(`API Error (${response.status}): ${JSON.stringify(errorData)}`);
  }

  const data = await response.json();
  return JSON.parse(data) as OutputParams;
};
