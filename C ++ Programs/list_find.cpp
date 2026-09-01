#include <iostream>
#include <list>
#include <algorithm>

using namespace std;

int main()
{
  list<int> l = {10, 20, 30, 40};

  auto it = std::find(l.begin(), l.end(), 30);
  if (it != l.end())
  {
    cout << "Found: " << *it << endl;

    *it = 35;
  }
  else
  {
    cout << "Element not found!" << endl;
  }
  for (int x : l)
  {
    cout << x << " ";
  }
  cout << endl;
  return 0;
}