import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'UrbanBuzz',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: HomePage(),
    );
  }
}

class HomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('UrbanBuzz'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => TrendingEventsPage()),
                );
              },
              child: Text('Trending Events'),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => TrendingFoodsPage()),
                );
              },
              child: Text('Trending Foods'),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => TrendingFashionPage()),
                );
              },
              child: Text('Trending Fashion'),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => TrendingTouristSpotsPage()),
                );
              },
              child: Text('Trending Tourist Spots'),
            ),
          ],
        ),
      ),
    );
  }
}

class TrendingEventsPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Trending Events'),
      ),
      body: Center(
        child: Text('List of Trending Events'),
      ),
    );
  }
}

class TrendingFoodsPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Trending Foods'),
      ),
      body: Center(
        child: Text('List of Trending Foods'),
      ),
    );
  }
}

class TrendingFashionPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Trending Fashion'),
      ),
      body: Center(
        child: Text('List of Trending Fashion'),
      ),
    );
  }
}

class TrendingTouristSpotsPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Trending Tourist Spots'),
      ),
      body: Center(
        child: Text('List of Trending Tourist Spots'),
      ),
    );
  }
}
