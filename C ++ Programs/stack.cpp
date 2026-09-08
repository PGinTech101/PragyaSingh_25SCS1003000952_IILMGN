#include <iostream>
#include <stack>
using namespace std;

int main()
{
  stack<int> s;
  s.push(5);
  s.push(9);
  s.push(8);
  s.push(4);
  s.push(10);
  while (!s.empty())
  {
    cout << s.top() << " ";
    s.pop();
  }
  cout << endl;
  return 0;
}