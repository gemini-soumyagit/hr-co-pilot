
// import { Client, Users } from 'node-appwrite';

// export default async ({ req, res, log, error }) => {
//   if (req.method == 'GET') {
//     log("welcome to era of AI !")
//     return res.send("welcome to era of AI !")
//   }
//   if (req.method == 'POST') {
//     return res.json({
//       'sendData1': req.body,
//       'sendData': "hello world",
//     })
//   }
// };


import { Client } from 'node-appwrite';
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { StateGraph, END } from "@langchain/langgraph";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import dotenv from 'dotenv';
import { StructuredTool } from "langchain/tools";
import { Document, VectorStoreIndex, serviceContextFromDefaults, Gemini } from "llamaindex";

dotenv.config();

// Initialize Appwrite client
const client = new Client()
    .setEndpoint(process.env.APPWRITE_ENDPOINT)
    .setProject(process.env.APPWRITE_PROJECT_ID)
    .setKey(process.env.APPWRITE_API_KEY);

// Tool definitions
class HRInputProcessingTool extends StructuredTool {
  name = 'HR Input Processing';
  description = 'Use this tool to process and categorize HR-related queries';
  schema = z.object({
    query: z.string().describe("The HR-related query to categorize"),
  });

  async _call({ query }) {
    const lowercaseInput = query.toLowerCase();
    if (lowercaseInput.includes('policy') || lowercaseInput.includes('handbook')) {
      return 'POLICY';
    } else if (lowercaseInput.includes('leave') || lowercaseInput.includes('vacation')) {
      return 'LEAVE';
    } else if (lowercaseInput.includes('salary') || lowercaseInput.includes('compensation')) {
      return 'COMPENSATION';
    } else if (lowercaseInput.includes('training') || lowercaseInput.includes('development')) {
      return 'TRAINING';
    } else {
      return 'GENERAL';
    }
  }
}

class PineconeQueryTool extends StructuredTool {
  name = 'Pinecone Query';
  description = 'Use this tool to query budget related information from Pinecone';
  schema = z.object({
    query: z.string().describe("The query to send to Pinecone"),
  });
  pinecone;

  constructor(pinecone) {
    super();
    this.pinecone = pinecone;
  }

  async _call({ query }) {
    try {
      const index = this.pinecone.Index('chatbot');
      const embeddings = new HuggingFaceTransformersEmbeddings();
      const queryEmbedding = await embeddings.embedQuery(query);
      const queryResponse = await index.query({
        vector: queryEmbedding,
        topK: 1,
        includeMetadata: true,
      });
      return (queryResponse.matches[0]?.metadata?.text) || 'No relevant information found.';
    } catch (error) {
      console.error('Error querying Pinecone:', error);
      return 'Error retrieving information from Pinecone';
    }
  }
}

class LlamaIndexTool extends StructuredTool {
  name = 'LlamaIndex Query';
  description = 'Use this tool to query recent HR updates from LlamaIndex';
  schema = z.object({
    query: z.string().describe("The query to send to LlamaIndex"),
  });
  index;

  constructor() {
    super();
  }

  async initialize() {
    const documents = await this.fetchDocumentsFromAppwrite();
    const serviceContext = serviceContextFromDefaults({
      llm: new Gemini({ apiKey: process.env.GOOGLE_GEMINI_API_KEY }),
    });
    this.index = await VectorStoreIndex.fromDocuments(documents, { serviceContext });
  }

  async fetchDocumentsFromAppwrite() {
    try {
      const storage = new Storage(client);
      const files = await storage.listFiles(process.env.APPWRITE_BUCKET_ID);
      const documents = [];

      for (const file of files.files) {
        const fileContent = await storage.getFileDownload(process.env.APPWRITE_BUCKET_ID, file.$id);
        const content = await fileContent.text();
        documents.push(new Document({ text: content, id_: file.$id }));
      }

      return documents;
    } catch (error) {
      console.error('Error fetching documents from Appwrite:', error);
      throw error;
    }
  }

  async _call({ query }) {
    try {
      if (!this.index) {
        await this.initialize();
      }
      const queryEngine = this.index.asQueryEngine();
      const response = await queryEngine.query(query);
      return response.toString();
    } catch (error) {
      console.error('Error querying LlamaIndex:', error);
      return 'Error retrieving information from LlamaIndex';
    }
  }
}

// HR Copilot setup
let hrCopilot = null;

async function setupPinecone() {
  return new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
}

async function createEnhancedHRCopilotAgent(pinecone) {
  const model = new ChatGoogleGenerativeAI({
    modelName: "gemini-pro",
    maxOutputTokens: 2048,
    temperature: 0,
  });

  const tools = [
    new HRInputProcessingTool(),
    new PineconeQueryTool(pinecone),
    new LlamaIndexTool(),
  ];

  const workflow = new StateGraph({
    channels: {
      query: z.string(),
      category: z.string(),
      pinecone_result: z.string(),
      llamaindex_result: z.string(),
      employee_context: z.object({
        department: z.string(),
        role: z.string(),
        tenure: z.string()
      }),
      conversation_history: z.array(z.object({
        role: z.enum(['user', 'assistant']),
        content: z.string()
      })),
      final_response: z.string(),
    }
  });

  // Node definitions (employeeContextNode, categorizeNode, pineconeNode, llamaindexNode, responseNode, conversationHistoryNode)
  // Employee Context Node
  const employeeContextNode = workflow.addNode("fetch_employee_context", async (state) => {
    return {
        employee_context: state.employee_context
    };
});

// Categorize Node
const categorizeNode = workflow.addNode("categorize", async (state) => {
    const tool = tools.find(t => t.name === 'HR Input Processing');
    const result = await tool._call({ query: state.query });
    return { category: result };
});

// Pinecone Query Node
const pineconeNode = workflow.addNode("pinecone_query", async (state) => {
    const tool = tools.find(t => t.name === 'Pinecone Query');
    const result = await tool._call({ query: state.query });
    return { pinecone_result: result };
});

// LlamaIndex Query Node
const llamaindexNode = workflow.addNode("llamaindex_query", async (state) => {
    const tool = tools.find(t => t.name === 'LlamaIndex Query');
    const result = await tool._call({ query: state.query });
    return { llamaindex_result: result };
});

// Response Node
const responseNode = workflow.addNode("respond", async (state) => {
    const responsePrompt = PromptTemplate.fromTemplate(
        `You are an HR assistant. Use the provided information to answer the user's query.

        User Query: {query}
        Category: {category}
        Pinecone Result: {pinecone_result}
        LlamaIndex Result: {llamaindex_result}
        Employee Context: {employee_context}
        Conversation History: {conversation_history}

        Please provide a structured and comprehensive response based on the above information, tailored to the employee's context. 
        
        Important: If information is available from either Pinecone or LlamaIndex, use it in your response. Do not state that you don't have access to information if it's provided in either the Pinecone or LlamaIndex results.
        
        If you truly don't have enough information from any source, politely explain that you don't have sufficient data and suggest contacting HR for more details.`
    );

    const chain = RunnableSequence.from([
        responsePrompt,
        model,
        new StringOutputParser(),
    ]);

    const result = await chain.invoke(state);
    return { final_response: result };
});

// Conversation History Node
const conversationHistoryNode = workflow.addNode("update_conversation_history", async (state) => {
    const updatedHistory = [
        ...state.conversation_history,
        { role: 'user', content: state.query },
        { role: 'assistant', content: state.final_response }
    ];
    return { conversation_history: updatedHistory };
});

  // Edge definitions
  workflow.setEntryPoint("fetch_employee_context");
  workflow.addEdge("fetch_employee_context", "categorize");
  workflow.addEdge("categorize", "pinecone_query");
  workflow.addEdge("pinecone_query", "llamaindex_query");
  workflow.addEdge("llamaindex_query", "respond");
  workflow.addEdge("respond", "update_conversation_history");
  workflow.addEdge("update_conversation_history", END);

  return workflow.compile();
}

async function initialize() {
  if (!hrCopilot) {
    const pinecone = await setupPinecone();
    hrCopilot = await createEnhancedHRCopilotAgent(pinecone);
  }
}

// Main Appwrite function
export default async function({ req, res, log, error }) {
  if (req.method !== 'POST') {
    return res.json({
      status: 405,
      message: 'Method Not Allowed'
    }, 405);
  }

  try {
    await initialize();

    const { query, employee_context, conversation_history } = req.body;

    const result = await hrCopilot.invoke({
      query,
      employee_context,
      conversation_history
    });

    return res.json({
      status: 200,
      message: "Success",
      final_response: result.final_response,
      category: result.category
    });
  } catch (err) {
    error('Error in HR Copilot:', err);
    return res.json({
      status: 500,
      message: 'Internal Server Error',
      error: err.message
    }, 500);
  }
}