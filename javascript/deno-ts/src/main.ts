console.log('Deno TypeScript 環境が正常に動作しています！');

import { serve } from "std/http/server.ts";

interface Todo {
  id: number;
  title: string;
  completed: boolean;
}

class TodoStore {
  private todos: Map<number, Todo> = new Map();
  private nextId = 1;

  addTodo(title: string): Todo {
    const todo: Todo = {
      id: this.nextId++,
      title,
      completed: false,
    };
    this.todos.set(todo.id, todo);
    console.log(`Todo 追加: ${title}`);
    return todo;
  }

  getTodos(): Todo[] {
    return Array.from(this.todos.values());
  }

  toggleTodo(id: number): boolean {
    const todo = this.todos.get(id);
    if (todo) {
      todo.completed = !todo.completed;
      return true;
    }
    return false;
  }
}

const store = new TodoStore();
store.addTodo('Deno の学習');
store.addTodo('TypeScript プロジェクトの作成');
store.addTodo('REST API の実装');

console.log('\n現在の Todo リスト:');
console.log(store.getTodos());

async function runServer() {
  const port = 8000;
  console.log(`\nHTTPサーバーを起動します: http://localhost:${port}/`);
  console.log('Ctrl+C で終了します');

  const handler = (req: Request): Response => {
    const url = new URL(req.url);
    
    if (url.pathname === '/') {
      return new Response(
        JSON.stringify({
          message: 'Deno TypeScript サーバーが動作中です！',
          todos: store.getTodos(),
          timestamp: new Date().toISOString(),
        }, null, 2),
        {
          headers: { 'content-type': 'application/json' },
        }
      );
    }
    
    return new Response('Not Found', { status: 404 });
  };

  await serve(handler, { port });
}

console.log('\nDeno の機能デモ:');
const textEncoder = new TextEncoder();
const data = textEncoder.encode('Hello from Deno!\\n');
await Deno.writeFile('./demo.txt', data);
console.log('ファイル書き込み完了: demo.txt');

const fileData = await Deno.readFile('./demo.txt');
const textDecoder = new TextDecoder();
console.log('ファイル読み込み結果:', textDecoder.decode(fileData));

await Deno.remove('./demo.txt');
console.log('ファイル削除完了');

if (import.meta.main) {
  console.log('\\nサーバーを起動するには、Ctrl+C で終了してから以下を実行してください:');
  console.log('deno task dev');
}