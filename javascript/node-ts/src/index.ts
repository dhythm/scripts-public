console.log('Node.js TypeScript 環境が正常に動作しています！');

interface User {
  id: number;
  name: string;
  email: string;
}

class UserService {
  private users: User[] = [];

  addUser(user: User): void {
    this.users.push(user);
    console.log(`ユーザー追加: ${user.name}`);
  }

  getUsers(): User[] {
    return this.users;
  }

  findUserById(id: number): User | undefined {
    return this.users.find(user => user.id === id);
  }
}

const userService = new UserService();

userService.addUser({ id: 1, name: '田中太郎', email: 'tanaka@example.com' });
userService.addUser({ id: 2, name: '佐藤花子', email: 'sato@example.com' });

console.log('登録ユーザー:', userService.getUsers());

const user = userService.findUserById(1);
if (user) {
  console.log(`ID 1 のユーザー: ${user.name} (${user.email})`);
}

async function fetchData(): Promise<void> {
  console.log('\n非同期処理のデモ:');
  await new Promise(resolve => setTimeout(resolve, 1000));
  console.log('1秒後に表示されます');
}

fetchData().then(() => {
  console.log('非同期処理完了');
});