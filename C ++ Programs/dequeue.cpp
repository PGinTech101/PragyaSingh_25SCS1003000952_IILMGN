#include <bits/stdc++.h>
using namespace std;

int main()
{
  deque<int> q;
  q.push_back(12);
  q.push_back(42);
  q.push_back(32);
  q.push_front(16);
  q.push_front(36);
  q.push_front(26);
  for (const auto &element : q)
  {
    cout << element << " ";
  }
  cout << endl;
  q.pop_front();
}