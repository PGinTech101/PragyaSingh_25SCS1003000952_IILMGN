#include <iostream>
#include <list>
using namespace std;

int main()
{
  list<int> l;
  list<int> l1 = {1, 3, 5};

  l.push_back(15);
  l.push_back(5);
  l.push_front(8);
  l.pop_back();
  l.pop_front();
  for (auto x : l)
  {
    cout << x << " ";
  }
  cout << "\n";
  auto it = l.begin();
  advance(it, 1);
  l.insert(it, 20);
  for (auto iter = l.begin(); iter != l.end(); iter++)
  {
    cout << *iter << " ";
  }
  cout << "\n";
  return 0;
}