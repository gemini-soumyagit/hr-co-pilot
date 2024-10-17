// main.js for Appwrite Cloud Function

import { Client } from 'node-appwrite';
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { Pinecone } from "@pinecone-database/pinecone";
import { StateGraph, END } from "@langchain/langgraph";
import { RunnableSequence } from "@langchain/core/runnables";
import { StructuredTool } from "langchain/tools";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import {
  Gemini,
  SimpleDirectoryReader,
  HuggingFaceEmbedding,
  VectorStoreIndex,
  Settings,
  GEMINI_MODEL,
} from "llamaindex";

// Pinecone setup and query function
async function setupPinecone() {
  const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
  return pinecone;
}

async function queryPinecone(pinecone, query) {
  try {
    const index = pinecone.Index('chatbot');
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

// HR Input Processing Tool
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
    } else {
      return 'GENERAL';
    }
  }
}

// Pinecone Query Tool
class PineconeQueryTool extends StructuredTool {
  name = 'Pinecone Query';
  description = 'Use this tool to query HR policies from Pinecone';
  schema = z.object({
    query: z.string().describe("The query to send to Pinecone"),
  });
  pinecone;

  constructor(pinecone) {
    super();
    this.pinecone = pinecone;
  }

  async _call({ query }) {
    return await queryPinecone(this.pinecone, query);
  }
}

// LlamaIndex Tool
class LlamaIndexTool extends StructuredTool {
  name = 'LlamaIndex Query';
  description = 'Use this tool to query recent HR updates from LlamaIndex';
  schema = z.object({
    query: z.string().describe("The query to send to LlamaIndex"),
  });
  
  constructor() {
    super();
    this.initializeLlamaIndex();
  }

  async initializeLlamaIndex() {
    if (!process.env.GOOGLE_API_KEY) {
      throw new Error("GOOGLE_API_KEY is not set in the environment variables");
    }

    Settings.llm = new Gemini({
      model: GEMINI_MODEL.GEMINI_PRO,
    });

    Settings.embedModel = new HuggingFaceEmbedding();

    // Note: In Appwrite, you might need to adjust how you load documents
    // This is just a placeholder and might need to be modified
    const documents = [{ text: "Sample HR document content" }];

    this.index = await VectorStoreIndex.fromDocuments(documents);
    this.queryEngine = this.index.asQueryEngine();
  }

  async _call({ query }) {
    try {
      const results = await this.queryEngine.query({
        query,
      });
      return results.message.content;
    } catch (error) {
      console.error('Error querying LlamaIndex:', error);
      return 'Error retrieving information from LlamaIndex';
    }
  }
}

// AI Agent Management
class AIAgentManager {
  constructor() {
    this.agents = new Map();
  }

  addAgent(name, type) {
    this.agents.set(name, { type, status: 'active', metrics: {} });
  }

  updateAgentMetrics(name, metrics) {
    const agent = this.agents.get(name);
    if (agent) {
      agent.metrics = { ...agent.metrics, ...metrics };
    }
  }

  getAgentStatus(name) {
    return this.agents.get(name)?.status || 'not found';
  }
}

// Ticket Management System
class TicketManagementSystem {
  constructor() {
    this.tickets = new Map();
  }

  createTicket(description, priority = 'medium') {
    const id = `TICKET-${Date.now()}`;
    this.tickets.set(id, { id, description, status: 'open', priority, updates: [] });
    return id;
  }

  updateTicket(id, update) {
    const ticket = this.tickets.get(id);
    if (ticket) {
      ticket.updates.push({ timestamp: new Date(), content: update });
    }
  }

  getTicketStatus(id) {
    return this.tickets.get(id)?.status || 'not found';
  }
}

// Enhanced HR Copilot Agent
async function createEnhancedHRCopilotAgent(pinecone, agentManager, ticketSystem) {
  const model = new ChatGoogleGenerativeAI({
    modelName: "gemini-pro",
    maxOutputTokens: 2048,
    temperature: 0,
    apiKey: process.env.GOOGLE_API_KEY,
  });

  const tools = [
    new HRInputProcessingTool(),
    new PineconeQueryTool(pinecone),
    new LlamaIndexTool(),
  ];

  // Enhanced state definition
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

  // Employee Context Node
  const employeeContextNode = workflow.addNode("fetch_employee_context", async (state) => {
    return {
      employee_context: {
        department: state.employee_context.department,
        role: state.employee_context.role,
        tenure: state.employee_context.tenure
      }
    };
  });

  // Categorize Node
  const categorizeNode = workflow.addNode("categorize", async (state) => {
    const tool = tools.find(t => t.name === 'HR Input Processing');
    const result = await tool._call({ query: state.query });
    agentManager.updateAgentMetrics('HR Input Processing', { queries_processed: 1 });
    return { category: result };
  });

  // Pinecone Query Node
  const pineconeNode = workflow.addNode("pinecone_query", async (state) => {
    const tool = tools.find(t => t.name === 'Pinecone Query');
    const result = await tool._call({ query: state.query });
    agentManager.updateAgentMetrics('Pinecone Query', { queries_processed: 1 });
    return { pinecone_result: result };
  });

  // LlamaIndex Query Node
  const llamaindexNode = workflow.addNode("llamaindex_query", async (state) => {
    const tool = tools.find(t => t.name === 'LlamaIndex Query');
    const result = await tool._call({ query: state.query });
    agentManager.updateAgentMetrics('LlamaIndex Query', { queries_processed: 1 });
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
  
      Please provide a comprehensive response based on the above information, tailored to the employee's context. 
      
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

// Appwrite function handler
export default async function appwriteFunction(req, res) {
  const client = new Client();

  // Don't forget to add your Appwrite endpoint and project ID in the Appwrite console
  client
    .setEndpoint('https://cloud.appwrite.io/v1')
    .setProject(process.env.APPWRITE_FUNCTION_PROJECT_ID)
    .setKey(process.env.APPWRITE_API_KEY);

  const pinecone = await setupPinecone();
  const agentManager = new AIAgentManager();
  agentManager.addAgent('HR Input Processing', 'categorization');
  agentManager.addAgent('Pinecone Query', 'knowledge_base');
  agentManager.addAgent('LlamaIndex Query', 'knowledge_base');

  const ticketSystem = new TicketManagementSystem();

  try {
    const hrCopilot = await createEnhancedHRCopilotAgent(pinecone, agentManager, ticketSystem);

    // Parse the request payload
    const { query, employeeContext } = JSON.parse(req.payload);

    const result = await hrCopilot.invoke({
      query: query,
      employee_context: employeeContext,
      conversation_history: []
    });

    // Handle ticket creation if necessary
    if (result.final_response.toLowerCase().includes("contact hr")) {
      const ticketId = ticketSystem.createTicket(`Query requires human assistance: ${query}`, "high");
      result.ticketId = ticketId;
    }

    // Send the response back
    res.json({
      result: result.final_response,
      category: result.category,
      pinecone_result: result.pinecone_result,
      llamaindex_result: result.llamaindex_result,
      employee_context: result.employee_context,
      conversation_history: result.conversation_history,
      ticketId: result.ticketId
    });
  } catch (error) {
    console.error('Error in Appwrite function:', error);
    res.json({
      error: 'An error occurred while processing your request.',
      details: error.message
    }, 500);
  }
}