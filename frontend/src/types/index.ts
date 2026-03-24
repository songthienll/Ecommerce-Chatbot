export interface Product {
  product_name: string;
  category: string;
  price: number;
  original_price?: number;
  url: string;
  thumbnail_url: string;
  rating: number;
  review_count: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatResponse {
  answer: string;
  sources: Product[];
}
