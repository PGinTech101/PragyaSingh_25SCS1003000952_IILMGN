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
  l.push_front(16);
  l.push_front(20);
  l.pop_back();
  l.pop_front();
  cout << "List: ";
  for (auto x : l)
  {
    cout << x << " ";
  }
  cout << endl;
  auto it = l.begin();
  advance(it, 1);
  l.insert(it, 20);
  cout << "New List: ";
  for (auto iter = l.begin(); iter != l.end(); iter++)
  {
    cout << *iter << " ";
  }
  cout << endl;
  return 0;
}