#include <bits/stdc++.h>
using namespace std;

int main()
{
  queue<int> q;
  q.push(12);
  q.push(15);
  q.push(75);
  q.push(85);
  q.push(95);
  for (; !q.empty(); q.pop())
  {
    cout << q.front() << " ";
  }
  cout << endl;
}