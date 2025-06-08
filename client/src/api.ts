import { OutputParams } from './params';

const server = 'http://127.0.0.1:5000';

export const decode = async (inputList: number[]): Promise<OutputParams | null> => {
  const response = await fetch(`${server}/decode?input=${JSON.stringify(inputList)}`);
  if (response.status === 400) {
    return null;
  }
  const data = await response.json();
  return JSON.parse(data) as OutputParams;
};
