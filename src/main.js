// main.js for Appwrite Cloud Function (ES6 Modules)

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

// Simplified Pinecone setup and query function
async function setupPinecone() {
  return new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
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

// Simplified HR Input Processing Tool
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

// Simplified Pinecone Query Tool
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

// Simplified AI Agent Management
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
}

// Simplified Ticket Management System
class TicketManagementSystem {
  constructor() {
    this.tickets = new Map();
  }

  createTicket(description, priority = 'medium') {
    const id = `TICKET-${Date.now()}`;
    this.tickets.set(id, { id, description, status: 'open', priority, updates: [] });
    return id;
  }
}

// Simplified HR Copilot Agent
async function createHRCopilotAgent(pinecone, agentManager) {
  const model = new ChatGoogleGenerativeAI({
    modelName: "gemini-pro",
    maxOutputTokens: 2048,
    temperature: 0,
    apiKey: process.env.GOOGLE_API_KEY,
  });

  const tools = [
    new HRInputProcessingTool(),
    new PineconeQueryTool(pinecone),
  ];

  const workflow = new StateGraph({
    channels: {
      query: z.string(),
      category: z.string(),
      pinecone_result: z.string(),
      final_response: z.string(),
    }
  });

  // Simplified nodes
  const categorizeNode = workflow.addNode("categorize", async (state) => {
    const tool = tools.find(t => t.name === 'HR Input Processing');
    const result = await tool._call({ query: state.query });
    agentManager.updateAgentMetrics('HR Input Processing', { queries_processed: 1 });
    return { category: result };
  });

  const pineconeNode = workflow.addNode("pinecone_query", async (state) => {
    const tool = tools.find(t => t.name === 'Pinecone Query');
    const result = await tool._call({ query: state.query });
    agentManager.updateAgentMetrics('Pinecone Query', { queries_processed: 1 });
    return { pinecone_result: result };
  });

  const responseNode = workflow.addNode("respond", async (state) => {
    const responsePrompt = PromptTemplate.fromTemplate(
      `You are an HR assistant. Use the provided information to answer the user's query.
  
      User Query: {query}
      Category: {category}
      Pinecone Result: {pinecone_result}
  
      Please provide a comprehensive response based on the above information.`
    );
  
    const chain = RunnableSequence.from([
      responsePrompt,
      model,
      new StringOutputParser(),
    ]);
  
    const result = await chain.invoke(state);
    return { final_response: result };
  });

  // Edge definitions
  workflow.setEntryPoint("categorize");
  workflow.addEdge("categorize", "pinecone_query");
  workflow.addEdge("pinecone_query", "respond");
  workflow.addEdge("respond", END);

  return workflow.compile();
}

// Appwrite function handler
export default async function(req, res) {
  const client = new Client();

  if (
    !process.env.APPWRITE_FUNCTION_ENDPOINT ||
    !process.env.APPWRITE_FUNCTION_API_KEY
  ) {
    console.warn("Environment variables are not set. Function cannot use Appwrite SDK.");
  } else {
    client
      .setEndpoint(process.env.APPWRITE_FUNCTION_ENDPOINT)
      .setProject(process.env.APPWRITE_FUNCTION_PROJECT_ID)
      .setKey(process.env.APPWRITE_FUNCTION_API_KEY);
  }

  const pinecone = await setupPinecone();
  const agentManager = new AIAgentManager();
  agentManager.addAgent('HR Input Processing', 'categorization');
  agentManager.addAgent('Pinecone Query', 'knowledge_base');

  const ticketSystem = new TicketManagementSystem();

  try {
    const hrCopilot = await createHRCopilotAgent(pinecone, agentManager);

    // Parse the request payload
    const { query } = JSON.parse(req.payload);

    const result = await hrCopilot.invoke({ query: query });

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