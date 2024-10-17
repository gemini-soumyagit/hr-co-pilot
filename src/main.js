import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { Pinecone } from "@pinecone-database/pinecone";
import { StateGraph, END } from "@langchain/langgraph";
import { RunnableSequence } from "@langchain/core/runnables";
import { StructuredTool } from "langchain/tools";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import dotenv from 'dotenv';
import {
  Gemini,
  Document,
  SimpleDirectoryReader,
  HuggingFaceEmbedding,
  VectorStoreIndex,
  Settings,
  GEMINI_MODEL,
  storageContextFromDefaults
} from "llamaindex";

dotenv.config();

export default async ({ req, res, log, error }) => {
  // if(req.method == 'GET'){

  //   return res.send("welcome to era of AI !")
  // }
  // if(req.method == 'POST'){

  //   return res.json({
  //     'sendData':req.body,
  //     'googleAPiKey': process.env.GOOGLE_GEMINI_API_KEY
  //   })
  // }
  try {
    // Initialize the embedding model
    const model = new HuggingFaceTransformersEmbeddings({
      model: "Xenova/all-MiniLM-L6-v2",
    });

    // Initialize Pinecone and specify the index
    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index('chatbot');

    // Create the PineconeStore from the existing index
    const vectorStores = await PineconeStore.fromExistingIndex(
      model,
      { pineconeIndex, }
    );

    // Initialize the chat model (Google Gemini)
    const chatModel = new ChatGoogleGenerativeAI({
      modelName: "gemini-1.5-pro",
      temperature: 0.7,
      maxRetries: 2,
      apiKey: process.env.GOOGLE_GEMINI_API_KEY
    });

    // Use the vector store as a retriever
    const vectorStoreRetriever = vectorStores.asRetriever();

    // Define the VectorStoreQATool
    class VectorStoreQATool extends Tool {
      name = "VectorStoreQA";
      description = "This tool MUST be used to access the budget document content. It retrieves relevant information from the document based on the input query.";

      constructor(retriever) {
        super();
        this.retriever = retriever;
      }

      async _call(input) {
        console.log("VectorStoreQATool called with input:", input);
        const results = await this.retriever.getRelevantDocuments(input);
        console.log("Retrieved Results:", results);

        if (results.length === 0) {
          return "No relevant information found in the budget document.";
        }

        return results.map(doc => doc.pageContent).join("\n\n");
      }
    }

    const vectorStoreQATool = new VectorStoreQATool(vectorStoreRetriever);
    const tools = [vectorStoreQATool];

    // Create the agent prompt
    const prompt = ChatPromptTemplate.fromMessages([
      ["system", `You are an AI assistant specifically designed to analyze and summarize a budget document.             Do not answer outside the pdf related question.
      The budget document is already loaded and accessible through the VectorStoreQA tool. 
      You MUST ALWAYS use the VectorStoreQA tool first to retrieve information before answering any question.
      DO NOT ask for the document to be provided - it is already available through the tool.
      After using the tool, summarize or answer based on the retrieved information.
      If the tool doesn't return any relevant information, say so clearly.`],
      ["human", "{input}"],
      new MessagesPlaceholder("agent_scratchpad"),
    ]);

    // Create the agent
    const agent = await createOpenAIFunctionsAgent({
      llm: chatModel,
      tools,
      prompt,
    });

    // Create an agent executor
    const agentExecutor = AgentExecutor.fromAgentAndTools({
      agent,
      tools,
      verbose: true,
    });

    console.log("Executing agent with input:", req.body.question);

    // Force the use of the VectorStoreQA tool for the first interaction
    const toolResult = await vectorStoreQATool._call(req.body.question);
    console.log("Forced tool usage result:", toolResult);

    // Execute the agent with the tool result
    const result = await agentExecutor.invoke({
      input: `Based on the following information from the budget document, ${req.body.question}\n\nDocument content: ${toolResult}`,
    });

    console.log("Agent execution result:", result);

    // Send the answer as a response
    res.json({ answer: result.output });
  } catch (error) {
    console.error('Error in ChatPdfController:', error);
    res.status(500).json({ error: "Internal Server Error", details: error.message });
  }
};
