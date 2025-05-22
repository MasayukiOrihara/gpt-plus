import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { Message as VercelChatMessage, LangChainAdapter } from 'ai';
 
// チャット形式
const formatMessage = (message: VercelChatMessage) => {
  return `${message.role}: ${message.content}`;
};

const PROMPT_TEMPLATE_TEXT = `"Current conversation:\n{history}\n\nuser:{input}\nAI:`;

/**
 * チャット応答AI
 * @param req 
 * @returns 
 */
export async function POST(req: Request) {
  try{
    const body = await req.json();
    const messages = body.messages ?? [];

    // 過去の履歴{chat_history}
    const formattedPreviousMessages = messages
      .slice(0, -1)
      .map(formatMessage)

    // メッセージ{input}
    const currentMessageContent = messages[messages.length - 1].content;

    // モデルの指定
    const model = new ChatOpenAI({
      apiKey: process.env.OPENAI_API_KEY!,
      model: 'gpt-4o',
      temperature: 0.8,
      });
        
    const prompt = PromptTemplate.fromTemplate(PROMPT_TEMPLATE_TEXT);
    const chain = prompt.pipe(model);

    const stream = await chain.stream({
      history: formattedPreviousMessages.join('\n'),
      input: currentMessageContent,
    });

    return LangChainAdapter.toDataStreamResponse(stream);
  } catch (error) {
    if (error instanceof Error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      });
    }
 
    return new Response(
      JSON.stringify({ error: 'Unknown error occurred' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } },
    );
  }
}